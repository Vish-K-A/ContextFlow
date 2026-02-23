import json
import os
import hashlib
import warnings
import logging
import numpy as np
import torch
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

warnings.filterwarnings("ignore", message=".*position_ids.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

_ = load_dotenv(find_dotenv())

_retriever_cache: dict = {}
_chain_cache: dict = {}
_cross_encoder_model = None
_llm = None
_openai_client = None

STORE_DIR = "faiss_store"

INTENT_CATEGORIES = {
    "summarize": (
        "The user wants to condense, shorten, or get an overview "
        "of text or a topic."
    ),
    "infer": (
        "The user wants sentiment analysis, topic extraction, "
        "entity recognition, or classification."
    ),
    "transform": (
        "The user wants translation, reformatting, converting to JSON/CSV, "
        "or changing writing style."
    ),
    "expand": (
        "The user wants to generate content from notes — emails, essays, "
        "blog posts, reports."
    ),
    "general": (
        "A general question or request that does not fit the above categories."
    ),
}

ROUTE_SYSTEM_PROMPTS = {
    "summarize": (
        "You are ContextFlow, an expert summarizer. "
        "Produce a clear, concise summary that preserves the key points. "
        "Structure the summary with a one-sentence TL;DR followed by the main points. "
        "Do not add information that is not present in the source material."
    ),
    "infer": (
        "You are ContextFlow, an analytical AI. "
        "Extract the information the user asks for — sentiment, topics, entities, "
        "or categories — and return it in a structured format with short explanations. "
        "Be precise and avoid speculation beyond what the text supports."
    ),
    "transform": (
        "You are ContextFlow, a transformation specialist. "
        "Convert or reformat the input exactly as the user requests. "
        "Preserve all factual content; change only structure, language, or format. "
        "If the target format is JSON or CSV, return only valid, parseable output."
    ),
    "expand": (
        "You are ContextFlow, a creative writing assistant. "
        "Expand the user's notes or outline into a full, polished piece of writing. "
        "Match the tone and style the user specifies. "
        "If no style is given, default to clear, professional prose."
    ),
    "general": (
        "You are ContextFlow, a helpful AI assistant. "
        "Answer the user's question clearly, concisely, and accurately. "
        "If you are unsure, say so rather than guessing. "
        "Think step by step before giving your final answer."
    ),
}

OUTPUT_EVAL_PROMPT = (
    "You are a response quality checker. "
    "Given the user's question and the AI's draft answer, "
    "evaluate whether the answer is: "
    "(1) accurate and relevant to the question, "
    "(2) safe and appropriate (no harmful, offensive, or misleading content), "
    "(3) complete — it does not leave the user's core question unanswered. "
    'Reply with a JSON object with two keys: "pass" (boolean) and "reason" (one sentence). '
    "If pass is false, the reason must explain what is wrong."
)

PDF_QA_SYSTEM = (
    "You are ContextFlow, a knowledgeable AI assistant. "
    "You have been given excerpts from a document to help answer the user's question. "
    "Use the excerpts as your primary source. "
    "If the excerpts do not fully answer the question, supplement with your own knowledge "
    "and clearly indicate which parts come from the document and which from general knowledge. "
    "Never simply say 'I don't know' — always provide the most helpful answer you can."
)

CONDENSE_TEMPLATE = (
    "Given the following conversation history and a follow-up question, "
    "rewrite the follow-up question to be a standalone question.\n\n"
    "Chat history:\n{chat_history}\n\n"
    "Follow-up question: {question}\n\n"
    "Standalone question:"
)


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def get_llm(model_name="gpt-4o-mini"):
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(temperature=0, model=model_name, max_tokens=1024)
    return _llm


def get_cross_encoder():
    global _cross_encoder_model
    if _cross_encoder_model is None:
        try:
            _cross_encoder_model = HuggingFaceCrossEncoder(
                model_name="BAAI/bge-reranker-base",
                model_kwargs={"device": "cpu"},
            )
            print("Cross-encoder loaded.")
        except Exception as exc:
            print(f"WARNING: Cross-encoder unavailable ({exc}). Reranking skipped.")
    return _cross_encoder_model


def moderate_input(text: str) -> dict:
    try:
        client = get_openai_client()
        result = client.moderations.create(input=text)
        output = result.results[0]
        flagged = output.flagged
        categories = {k: v for k, v in output.categories.__dict__.items() if v}
        return {"flagged": flagged, "categories": categories}
    except Exception as exc:
        print(f"Moderation API error: {exc}")
        return {"flagged": False, "categories": {}}


def detect_prompt_injection(text: str) -> bool:
    injection_patterns = [
        "ignore previous instructions",
        "ignore all instructions",
        "disregard your system prompt",
        "forget your instructions",
        "you are now",
        "act as if",
        "pretend you are",
        "your new instructions",
        "override your",
    ]
    lowered = text.lower()
    return any(pattern in lowered for pattern in injection_patterns)


def classify_intent(question: str) -> str:
    llm = get_llm()
    category_descriptions = "\n".join(
        f'- "{k}": {v}' for k, v in INTENT_CATEGORIES.items()
    )
    classification_prompt = (
        "Classify the user's request into exactly one of these categories:\n"
        f"{category_descriptions}\n\n"
        f'User request: "{question}"\n\n'
        "Respond with only the category name, nothing else."
    )
    messages = [
        SystemMessage(content="You are an intent classifier. Reply with only the category name."),
        HumanMessage(content=classification_prompt),
    ]
    result = llm.invoke(messages).content.strip().lower()
    return result if result in INTENT_CATEGORIES else "general"


def evaluate_output(question: str, answer: str) -> dict:
    try:
        llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=256)
        eval_input = f"User question: {question}\n\nDraft answer: {answer}"
        messages = [
            SystemMessage(content=OUTPUT_EVAL_PROMPT),
            HumanMessage(content=eval_input),
        ]
        raw = llm.invoke(messages).content.strip()
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except Exception as exc:
        print(f"Output evaluation error: {exc}")
        return {"pass": True, "reason": "Evaluation skipped due to error."}


def answer_without_pdf(question: str, chat_history: list) -> str:
    moderation = moderate_input(question)
    if moderation["flagged"]:
        flagged_cats = ", ".join(moderation["categories"].keys())
        return (
            "I'm sorry, but I can't respond to that request. "
            f"It was flagged for: {flagged_cats}."
        )

    if detect_prompt_injection(question):
        return (
            "I detected an attempt to override my instructions. "
            "Please ask a genuine question and I'll be happy to help."
        )

    intent = classify_intent(question)
    system_prompt = ROUTE_SYSTEM_PROMPTS[intent]

    llm = get_llm()
    messages = [SystemMessage(content=system_prompt)]
    for human, ai in chat_history:
        messages.append(HumanMessage(content=human))
        messages.append(AIMessage(content=ai))
    messages.append(HumanMessage(content=question))

    draft_answer = llm.invoke(messages).content

    eval_result = evaluate_output(question, draft_answer)
    if not eval_result.get("pass", True):
        fallback_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    "The following draft answer was flagged as insufficient:\n\n"
                    f"Draft: {draft_answer}\n"
                    f"Issue: {eval_result.get('reason', '')}\n\n"
                    f"Please provide a corrected, complete answer to: {question}"
                )
            ),
        ]
        draft_answer = llm.invoke(fallback_messages).content

    return draft_answer


class AdapteredEmbeddings(Embeddings):
    def __init__(self, base_embeddings, adapter_matrix):
        self.base_embeddings = base_embeddings
        self.adapter_matrix = adapter_matrix

    def embed_documents(self, texts):
        return [self._apply_matrix(v) for v in self.base_embeddings.embed_documents(texts)]

    def embed_query(self, text):
        return self._apply_matrix(self.base_embeddings.embed_query(text))

    def _apply_matrix(self, vec):
        v = np.array(vec)
        v /= np.linalg.norm(v) + 1e-8
        projected = np.dot(self.adapter_matrix, v)
        return (projected / (np.linalg.norm(projected) + 1e-8)).tolist()


def setup_vectorstore(file_path: str, adapter_matrix=None):
    if file_path in _retriever_cache:
        print("Cache hit — retriever ready.")
        return _retriever_cache[file_path]

    doc_hash = hashlib.md5(file_path.encode()).hexdigest()[:12]
    index_dir = os.path.join(STORE_DIR, doc_hash)
    index_file = os.path.join(index_dir, "index.faiss")

    base_embeddings = OpenAIEmbeddings(chunk_size=500)
    final_embeddings = (
        AdapteredEmbeddings(base_embeddings, adapter_matrix)
        if adapter_matrix is not None
        else base_embeddings
    )

    if os.path.exists(index_file):
        print(f"Loading existing FAISS index from {index_dir}")
        vectorstore = FAISS.load_local(
            index_dir,
            final_embeddings,
            allow_dangerous_deserialization=True,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        _retriever_cache[file_path] = retriever
        return retriever

    print(f"Indexing: {file_path}")
    loader = PyPDFLoader(file_path)
    try:
        docs = loader.load()
    except Exception as exc:
        raise RuntimeError(f"Failed to load PDF: {exc}") from exc

    print(f"  {len(docs)} pages loaded.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    os.makedirs(index_dir, exist_ok=True)

    vectorstore = None
    batch_size = 100

    for i in range(0, len(docs), batch_size):
        page_batch = docs[i : i + batch_size]
        chunks = splitter.split_documents(page_batch)

        if not chunks:
            continue

        if vectorstore is None:
            vectorstore = FAISS.from_documents(chunks, final_embeddings)
        else:
            batch_vs = FAISS.from_documents(chunks, final_embeddings)
            vectorstore.merge_from(batch_vs)
            del batch_vs

        del chunks
        print(f"  Indexed {min(i + batch_size, len(docs))}/{len(docs)} pages")

    vectorstore.save_local(index_dir)
    print(f"  Saved FAISS index to {index_dir}")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    _retriever_cache[file_path] = retriever
    return retriever


def _condense_question(question: str, chat_history: list, llm) -> str:
    if not chat_history:
        return question
    history_str = "\n".join(
        f"Human: {h}\nAssistant: {a}" for h, a in chat_history
    )
    prompt = CONDENSE_TEMPLATE.format(chat_history=history_str, question=question)
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()


def _rerank_docs(docs: list, question: str, top_n: int = 3) -> list:
    cross_encoder = get_cross_encoder()
    if cross_encoder is None:
        return docs[:top_n]
    try:
        pairs = [(question, doc.page_content) for doc in docs]
        scores = cross_encoder.score(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_n]]
    except Exception as exc:
        print(f"WARNING: Reranker failed ({exc}). Using top-k docs directly.")
        return docs[:top_n]


class _Chain:
    def __init__(self, base_retriever, llm, top_n):
        self._retriever = base_retriever
        self._llm = llm
        self._top_n = top_n

    def invoke(self, inputs: dict) -> dict:
        return self(inputs)

    def __call__(self, inputs: dict) -> dict:
        question = inputs.get("question", "")
        chat_history = inputs.get("chat_history", [])

        moderation = moderate_input(question)
        if moderation["flagged"]:
            flagged_cats = ", ".join(moderation["categories"].keys())
            return {
                "answer": (
                    "I'm sorry, but I can't respond to that request. "
                    f"It was flagged for: {flagged_cats}."
                ),
                "source_documents": [],
            }

        if detect_prompt_injection(question):
            return {
                "answer": (
                    "I detected an attempt to override my instructions. "
                    "Please ask a genuine question and I'll be happy to help."
                ),
                "source_documents": [],
            }

        standalone_q = _condense_question(question, chat_history, self._llm)
        raw_docs = self._retriever.invoke(standalone_q)
        top_docs = _rerank_docs(raw_docs, standalone_q, top_n=self._top_n)

        context = "\n\n---\n\n".join(
            f"[Source: page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in top_docs
        )

        messages = [SystemMessage(content=PDF_QA_SYSTEM)]
        for human, ai in chat_history:
            messages.append(HumanMessage(content=human))
            messages.append(AIMessage(content=ai))
        messages.append(
            HumanMessage(
                content=(
                    f"Document excerpts:\n{context}\n\n"
                    f"Question: {standalone_q}"
                )
            )
        )

        draft_answer = self._llm.invoke(messages).content

        eval_result = evaluate_output(question, draft_answer)
        if not eval_result.get("pass", True):
            retry_messages = [
                SystemMessage(content=PDF_QA_SYSTEM),
                HumanMessage(
                    content=(
                        "The following draft answer was flagged as insufficient:\n\n"
                        f"Draft: {draft_answer}\n"
                        f"Issue: {eval_result.get('reason', '')}\n\n"
                        f"Document excerpts:\n{context}\n\n"
                        f"Please provide a corrected, complete answer to: {standalone_q}"
                    )
                ),
            ]
            draft_answer = self._llm.invoke(retry_messages).content

        return {"answer": draft_answer, "source_documents": top_docs}


def get_advanced_chain(base_retriever, model_name="gpt-4o-mini", top_n=3):
    cache_key = id(base_retriever)
    if cache_key in _chain_cache:
        return _chain_cache[cache_key]

    llm = get_llm(model_name)
    chain = _Chain(base_retriever, llm, top_n)
    _chain_cache[cache_key] = chain
    return chain


def train_embedding_adaptor(
    dataset, embedding_dim=1536, epochs=100, lr=0.001
):
    adapter_matrix = torch.randn(embedding_dim, embedding_dim, requires_grad=True)
    optimizer = torch.optim.Adam([adapter_matrix], lr=lr)
    criterion = torch.nn.MSELoss()
    cos_sim = torch.nn.CosineSimilarity(dim=0)

    for epoch in range(epochs):
        total_loss = 0.0
        for query_emb, doc_emb, relevance_score in dataset:
            q = torch.tensor(query_emb, dtype=torch.float32)
            d = torch.tensor(doc_emb, dtype=torch.float32)
            target = torch.tensor([relevance_score], dtype=torch.float32)
            uq = torch.nn.functional.normalize(
                torch.matmul(adapter_matrix, q), dim=0
            )
            d = torch.nn.functional.normalize(d, dim=0)
            sim = cos_sim(uq, d).unsqueeze(0)
            loss = criterion(sim, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            avg = total_loss / len(dataset)
            print(f"Epoch {epoch}: Loss {avg:.4f}")

    return adapter_matrix.detach().numpy()
