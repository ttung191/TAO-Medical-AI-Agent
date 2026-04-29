import os
import chromadb
from chromadb.utils import embedding_functions

class ClinicalVectorDB:
    def __init__(self, db_path="./data/vector_db"):
        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        # BAAI/bge-m3 hỗ trợ tiếng Việt rất tốt, model sẽ tự tải lần đầu chạy
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-m3")
        self.collection = self.client.get_or_create_collection(
            name="clinical_guidelines", 
            embedding_function=self.ef
        )

    def search(self, query: str, top_k: int = 3) -> list:
        if self.collection.count() == 0: return []
        res = self.collection.query(query_texts=[query], n_results=top_k)
        if not res['documents']: return []
        
        return [{"text": doc, "source": meta.get("source", "N/A")} 
                for doc, meta in zip(res['documents'][0], res['metadatas'][0])]