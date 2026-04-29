import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from pypdf import PdfReader
from tao_medical_ai.infrastructure.evidence.vector_db import ClinicalVectorDB

def ingest(data_dir="./data/guidelines"):
    os.makedirs(data_dir, exist_ok=True)
    db = ClinicalVectorDB()
    files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    
    if not files:
        print(f"Bỏ file PDF y khoa vào {data_dir} rồi chạy lại nhé!")
        return

    for file in files:
        print(f"Đang nạp: {file}...")
        path = os.path.join(data_dir, file)
        # Trích xuất text từ PDF
        text = "".join(page.extract_text() for page in PdfReader(path).pages if page.extract_text())
        
        # Cắt nhỏ (chunking) 1000 ký tự, overlap 200
        chunks = [text[i:i+1000] for i in range(0, len(text), 800)]
        
        db.collection.add(
            documents=chunks,
            metadatas=[{"source": file} for _ in chunks],
            ids=[f"{file}_chunk_{i}" for i in range(len(chunks))]
        )
        print(f"-> Nạp xong {len(chunks)} chunks.")

if __name__ == "__main__":
    ingest()