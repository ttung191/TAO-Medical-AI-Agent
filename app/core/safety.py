import re
from app.models.enums import RiskLevel

class SafetyFilter:
    def __init__(self):
        # Các từ khóa cờ đỏ (Red Flags) tuyệt đối
        self.critical_keywords = [
            r"suicide", r"kill myself", r"overdose", r"tự tử", r"tự sát", 
            r"uống thuốc ngủ", r"nhảy lầu", r"weapon", r"bomb"
        ]
        
    def assess_input_risk(self, text: str) -> RiskLevel:
        """Kiểm tra nhanh đầu vào của người dùng"""
        text_lower = text.lower()
        
        for pattern in self.critical_keywords:
            if re.search(pattern, text_lower):
                return RiskLevel.CRITICAL
                
        # Có thể mở rộng thêm logic kiểm tra độ dài, spam, v.v.
        return RiskLevel.LOW