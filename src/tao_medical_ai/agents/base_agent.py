from typing import List, Dict, Any


class BaseAgent:
    def __init__(self, role: str, tier: int, llm):
        self.role = role
        self.tier = tier
        self.llm = llm

    async def assess(self, case, prior_opinions: List[Dict[str, Any]] = None):
        prior_opinions = prior_opinions or []

        prompt = self._build_prompt(case, prior_opinions)

        response = await self.llm.complete(prompt)

        return {
            "role": self.role,
            "tier": self.tier,
            "risk": self._extract_risk(response.text),
            "confidence": self._extract_confidence(response.text),
            "reasoning": response.text,
            "cost": getattr(response, "cost", 0.0)
        }

    def _build_prompt(self, case, prior_opinions):
        prior_summary = "\n".join([f"  - {op['role']} (Tier {op['tier']}): {op.get('reasoning', '')[:200]}..." for op in prior_opinions]) if prior_opinions else "  Chưa có"
        
        return f"""
# HỆ THỐNG TAO - TRIAGE & ASSESSMENT ORCHESTRATION

Vai trò của bạn: **{self.role}** (Tier {self.tier})
Mục tiêu: Đánh giá mức độ rủi ro và quyết định escalation cho bệnh án

## BỆNH ÁN
- Lý do khám: {case.chief_complaint}
- Tuổi/Giới: {case.patient_age}/{case.patient_sex}
- Triệu chứng: {', '.join(case.symptoms[:5])}
- Tiền sử: {', '.join(case.medical_history[:3])}
- Sinh hiệu: {case.vitals}
- Xét nghiệm: {case.labs}

## LỊCH SỬ TUYẾN DƯỚI
{prior_summary}

## HƯỚNG DẪN QUYẾT ĐỊNH

### CẤP ĐỘ RỦI RO:
- **CRITICAL**: Đe dọa tính mạng ngay tức khắc (sốc, suy hô hấp, xuất huyết nặng)
- **HIGH**: Nguy hiểm nếu không can thiệp nhanh (ACS, sốc, ngạt, viêm ngoài màng não)
- **MODERATE**: Cần chẩn đoán & điều trị nhưng không tức thời (bệnh nhiễm khuẩn, rối loạn khí huyết)
- **LOW**: Không đe dọa mạng sống, có thể xử trí ngoại trú

### ESCALATION LOGIC:
- **escalate** nếu: Risk >= HIGH hoặc không chắc chắn → tuyến trên
- **review**: Risk = MODERATE + cần thêm dữ liệu/bác sĩ cùng tuyến xem lại
- **stop**: Risk = LOW + xác định → tuyến hiện tại/ngoại trú

## YÊU CẦU OUTPUT (JSON):
{{
  "risk": "low|moderate|high|critical",
  "escalation": "stop|review|escalate",
  "confidence": 0.0-1.0,
  "working_diagnosis": "Chẩn đoán dự phòng chính",
  "rationale": "Giải thích chi tiết quyết định",
  "clinical_reasoning": "Chuỗi suy luận: Triệu chứng X + Y → lo ngại Z → cần escalate vì..."
}}
"""

    def _extract_risk(self, text: str) -> str:
        """Extract risk level từ JSON response"""
        try:
            import json
            data = json.loads(text)
            risk = data.get("risk", "low").lower()
            return risk if risk in ["critical", "high", "moderate", "low"] else "low"
        except:
            text = text.lower()
            if "critical" in text:
                return "critical"
            if "high" in text:
                return "high"
            if "moderate" in text or "medium" in text:
                return "moderate"
            return "low"

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score từ JSON response"""
        try:
            import json
            import re
            data = json.loads(text)
            conf = float(data.get("confidence", 0.7))
            return max(0.0, min(1.0, conf))  # Clamp to [0,1]
        except:
            # Fallback: tìm pattern 0.X
            match = re.search(r'0\.[0-9]+', text)
            if match:
                return float(match.group())
            return 0.7