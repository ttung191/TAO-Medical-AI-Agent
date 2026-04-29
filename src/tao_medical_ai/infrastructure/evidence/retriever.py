from tao_medical_ai.infrastructure.evidence.vector_db import ClinicalVectorDB
from tao_medical_ai.contracts.case import CaseRequest

class RAGEvidenceRetriever:
    def __init__(self):
        self.db = ClinicalVectorDB()

    def retrieve(self, case: CaseRequest) -> str:
        symptoms = ", ".join(case.symptoms) if hasattr(case, 'symptoms') else ""
        history = case.medical_history if hasattr(case, 'medical_history') else ""
        
        query = f"Triệu chứng: {symptoms}. Tiền sử: {history}. Hướng dẫn phác đồ, tương tác và chống chỉ định."
        results = self.db.search(query)
        
        if not results: 
            return "[Cảnh báo: Không tìm thấy y văn nội bộ. Hãy dùng kiến thức cơ sở cẩn thận]"
            
        context = "--- TÀI LIỆU Y KHOA RAG (ƯU TIÊN TUÂN THỦ) ---\n"
        for idx, r in enumerate(results, 1):
            context += f"[{idx}] Nguồn {r['source']}:\n{r['text']}\n\n"
        return context