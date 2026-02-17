import os
import numpy as np
import torch
from typing import List, Any
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever, ContextualCompressionRetriever, MultiQueryRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.stores import InMemoryStore
from langchain_core.embeddings import Embeddings

_ = load_dotenv(find_dotenv())

class AdapteredEmbeddings(Embeddings):

    def __init__(self, base_embeddings: Embeddings, adapter_matrix: np.ndarray):
        self.base_embeddings = base_embeddings
        self.adapter_matrix = adapter_matrix
        if adapter_matrix.shape[0] != adapter_matrix.shape[1]:
            print(f"Warning: Non-square adapter matrix {adapter_matrix.shape}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raw_embeddings = self.base_embeddings.embed_documents(texts)
        return [self._apply_matrix(vec) for vec in raw_embeddings]

    def embed_query(self, text: str) -> List[float]:
        raw_embedding = self.base_embeddings.embed_query(text)
        return self._apply_matrix(raw_embedding)

    def _apply_matrix(self, vec: List[float]) -> List[float]:
        vec_np = np.array(vec)
        vec_np = vec_np / (np.linalg.norm(vec_np) + 1e-8)
        projected = np.dot(self.adapter_matrix, vec_np)
        projected = projected / (np.linalg.norm(projected) + 1e-8)
        return projected.tolist()

import hashlib

_retriever_cache: dict = {}

def setup_vectorstore(file_path, adapter_matrix=None):
    if file_path in _retriever_cache:
        return _retriever_cache[file_path]

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    base_embeddings = OpenAIEmbeddings()
    if adapter_matrix is not None:
        final_embeddings = AdapteredEmbeddings(base_embeddings, adapter_matrix)
    else:
        final_embeddings = base_embeddings

    doc_hash = hashlib.md5(file_path.encode()).hexdigest()[:12]
    collection_name = f"doc_{doc_hash}"

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=final_embeddings
    )

    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )

    retriever.add_documents(docs, ids=None)

    _retriever_cache[file_path] = retriever
    return retriever

def get_advanced_chain(base_retriever, model_name="gpt-4o-mini", top_n=5):
    llm = ChatOpenAI(temperature=0, model=model_name)

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, 
        llm=llm
    )

    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=top_n)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=multi_query_retriever
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=compression_retriever, 
        return_source_documents=True,
        verbose=True
    )
    
    return qa_chain

def train_embedding_adaptor(dataset, embedding_dim=1536, epochs=100, lr=0.001):

    adapter_matrix = torch.randn(embedding_dim, embedding_dim, requires_grad=True)
    optimizer = torch.optim.Adam([adapter_matrix], lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for query_emb, doc_emb, relevance_score in dataset:
            q_tensor = torch.tensor(query_emb, dtype=torch.float32)
            d_tensor = torch.tensor(doc_emb, dtype=torch.float32)
            target = torch.tensor([relevance_score], dtype=torch.float32)
            
            updated_query = torch.matmul(adapter_matrix, q_tensor)
            updated_query = torch.nn.functional.normalize(updated_query, dim=0)
            d_tensor = torch.nn.functional.normalize(d_tensor, dim=0)
            cosine_sim = torch.nn.functional.cosine_similarity(
                updated_query.unsqueeze(0), 
                d_tensor.unsqueeze(0)
            )
            
            loss = criterion(cosine_sim, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Avg Loss {total_loss/len(dataset):.4f}")

    return adapter_matrix.detach().numpy()