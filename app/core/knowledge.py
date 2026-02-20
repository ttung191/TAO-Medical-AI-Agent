import json
import os
from typing import List

class KnowledgeBase:
    def __init__(self):
        # Tự động load dữ liệu khi khởi tạo
        self.drug_data = self._load_json("data/drug_interactions.json")
        self.guidelines = self._load_json("data/medical_guidelines.json")

    def _load_json(self, path: str) -> dict:
        """Hàm đọc file JSON an toàn"""
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading {path}: {e}")
        return {}

    def check_drug_interactions(self, text: str) -> List[str]:
        """Quét văn bản tìm cặp thuốc tương tác"""
        warnings = []
        text_lower = text.lower()
        
        if not self.drug_data: return []

        for item in self.drug_data.get("interactions", []):
            pair = item["pair"]
            # Logic: Nếu cả 2 tên thuốc đều xuất hiện trong triệu chứng/tiền sử
            if all(drug.lower() in text_lower for drug in pair):
                warnings.append(
                    f"⚠️ DRUG INTERACTION ALERT: {pair[0]} + {pair[1]} -> "
                    f"{item['condition']} ({item['severity']}). Mechanism: {item['mechanism']}"
                )
        return warnings

    def get_medical_guidelines(self, text: str) -> List[str]:
        """Tìm phác đồ điều trị liên quan"""
        relevant_protocols = []
        text_lower = text.lower()
        
        if not self.guidelines: return []

        for proto in self.guidelines.get("protocols", []):
            # Tìm kiếm keyword đơn giản (condition name)
            if proto["condition"].lower() in text_lower:
                steps = "\n".join([f"- {step}" for step in proto["immediate_action"]])
                relevant_protocols.append(
                    f"📋 GUIDELINE FOR {proto['condition'].upper()}:\n"
                    f"Criteria: {proto['criteria']}\n"
                    f"Immediate Actions:\n{steps}"
                )
        return relevant_protocols

# Tạo một instance duy nhất để dùng chung cho toàn app
knowledge_engine = KnowledgeBase()