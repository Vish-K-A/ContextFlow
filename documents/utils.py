import os
import numpy as np
import torch
from typing import List, Any
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import ParentDocumentRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors import CrossEncoderReranker
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.stores import InMemoryStore
from langchain_community.retrievers import MultiQueryRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.embeddings import Embeddings

_ = load_dotenv(find_dotenv())

class AdapteredEmbeddings(Embeddings):

    def __init__(self, base_embeddings: Embeddings, adapter_matrix: np.ndarray):
        self.base_embeddings = base_embeddings
        self.adapter_matrix = adapter_matrix

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raw_embeddings = self.base_embeddings.embed_documents(texts)
        return [self._apply_matrix(vec) for vec in raw_embeddings]

    def embed_query(self, text: str) -> List[float]:
        raw_embedding = self.base_embeddings.embed_query(text)
        return self._apply_matrix(raw_embedding)

    def _apply_matrix(self, vec: List[float]) -> List[float]:
        vec_np = np.array(vec)
        projected = np.dot(self.adapter_matrix, vec_np)
        return projected.tolist()

def setup_vectorstore(file_path, adapter_matrix=None):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    base_embeddings = OpenAIEmbeddings()
    if adapter_matrix is not None:
        final_embeddings = AdapteredEmbeddings(base_embeddings, adapter_matrix)
    else:
        final_embeddings = base_embeddings

    vectorstore = Chroma(
        collection_name="full_rag_pipeline",
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
    return retriever

def get_advanced_chain(base_retriever, model_name="gpt-3.5-turbo"):
    
    llm = ChatOpenAI(temperature=0, model=model_name)

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, 
        llm=llm
    )

    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=5)
    
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

def train_embedding_adaptor(dataset):

    mat_size = len(dataset[0][0])
    adapter_matrix = torch.randn(mat_size, mat_size, requires_grad=True)
    optimizer = torch.optim.Adam([adapter_matrix], lr=0.01)
    criterion = torch.nn.MSELoss()

    for epoch in range(100):
        total_loss = 0
        for query_emb, doc_emb, label in dataset:
            q_tensor = torch.tensor(query_emb, dtype=torch.float32)
            d_tensor = torch.tensor(doc_emb, dtype=torch.float32)
            lbl_tensor = torch.tensor(label, dtype=torch.float32)
            
            updated_query = torch.matmul(adapter_matrix, q_tensor)
            
            cosine_sim = torch.nn.functional.cosine_similarity(updated_query.unsqueeze(0), d_tensor.unsqueeze(0))
            
            loss = criterion(cosine_sim, lbl_tensor)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch}: Loss {total_loss}")

    return adapter_matrix.detach().numpy()