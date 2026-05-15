import json, logging
from tao_medical_ai.infrastructure.llm.provider import LLMFactory
from tao_medical_ai.contracts.case import StructuredCase, AgentAssessment, TokenUsage
from tao_medical_ai.contracts.enums import RiskLevel, EscalationDecision, Disposition
from tao_medical_ai.security.phi_filter import PHIFilter

logger = logging.getLogger(__name__)

class DynamicMedicalAgent:
    def __init__(self, tier: int, role: str):
        self.tier = tier
        self.role = role
        self.llm_provider = LLMFactory.get_provider(self.tier)

    async def run(self, case: StructuredCase, history: dict) -> AgentAssessment:
        prior_notes = history.get('prior_notes', 'Chưa có')
        prior_risk = history.get('prior_risk', 'Chưa có')
        
        prompt = f"""# HỆ THỐNG TAO - TRIAGE & ASSESSMENT ORCHESTRATION v2

## VỊ TRÍ CỦA BẠN
- **Vai trò**: {self.role}
- **Tuyến**: {self.tier}
- **Mục tiêu**: Đánh giá mức độ rủi ro và quyết định escalation

## BỆNH ÁN
{case.to_agent_prompt()}

## LỊCH SỬ TỪ TUYẾN DƯỚI
- Đánh giá trước: {prior_risk}
- Ghi chú: {prior_notes}

## HƯỚNG DẪN QUYẾT ĐỊNH (Decision Tree)

### 1. PHÂN LOẠI RỦI RO
**CRITICAL** → Đe dọa tính mạng ngay (sốc, suy hô hấp, xuất huyết, ngạt, cơn hen nặng)
- Triệu chứng: Mất ý thức, SpO2 < 85%, BP < 90/60, HR > 140, chảy máu liên tục
- Hành động: ESCALATE ngay lập tức

**HIGH** → Nguy hiểm nếu không can thiệp sớm (ACS, sốc nhiễm khuẩn, viêm ngoài màng não, chấn thương đầu)
- Triệu chứng: Đau ngực / khó thở / sốt cao / đau đầu /thần kinh bất thường
- Hành động: ESCALATE → bác sĩ tuyến trên hoặc ED

**MODERATE** → Cần chẩn đoán & điều trị nhưng không tức thời (viêm phế quản, cảm cúm, viêm dạ dày, tây tư)
- Triệu chứng: Sốt < 38.5°C, ho, buồn nôn, đau nhẹ
- Hành động: REVIEW (cân cứ cơ sở vật chất) hoặc escalate nếu không chắc chắn

**LOW** → Không đe dọa mạng sống, tự chăm sóc/ngoại trú (cảm lạnh, bệnh ngoài da nhẹ, mệt)
- Triệu chứng: Sốt < 38°C, ho khô, ngứa
- Hành động: STOP → hướng dẫn tự chăm sóc hoặc tái khám

### 2. LOGIC ESCALATION
- **ESCALATE** nếu: Risk ≥ HIGH hoặc không chắc chắn hoặc cần chuyên gia
- **REVIEW** nếu: Risk = MODERATE + cần thêm dữ liệu/xét nghiệm + bác sĩ cùng tuyến xem lại
- **STOP** nếu: Risk = LOW + xác định được nguyên nhân + khả năng điều trị ngoại trú

### 3. DISPOSITION MAPPING
- **emergency_department**: Critical/High + triệu chứng cấp cứu
- **urgent_care**: High + có khả năng xử trí cấp cứu cơ bản
- **primary_care**: Moderate + có bác sĩ/phòng khám chuyên khoa
- **human_review**: Không chắc chắn hoặc cần quyết định bác sĩ
- **self_care**: Low + hướng dẫn chi tiết

### 4. SAFETY CONSTRAINTS
⚠️ LUÔN ESCALATE NẾU:
- Bệnh nhân > 65 tuổi + sốt hoặc khó thở
- Bệnh nhân < 6 tuổi + bất kỳ triệu chứng nặng
- Mang thai + bất kỳ triệu chứng lạ
- Dùng thuốc chống đông/corticoid + chảy máu
- Không chắc chắn → "human_review" an toàn hơn

## OUTPUT (JSON STRICT FORMAT):
{{
  "working_diagnosis": "Chẩn đoán dự phòng chính (VD: Viêm phế quản cấp tính)",
  "risk": "low|moderate|high|critical",
  "escalation": "stop|review|escalate",
  "suggested_disposition": "self_care|primary_care|urgent_care|emergency_department|human_review",
  "rationale": "Giải thích ngắn gọn (50-100 từ) - TẠI SAO risk + escalation decision này?",
  "clinical_reasoning": "Chuỗi suy luận: [Triệu chứng A] + [Triệu chứng B] → lo ngại [Chẩn đoán X] → escalate vì [Lý do] → disposition [Y]",
  "confidence": 0.85
}}

**IMPORTANT**: Chỉ trả JSON, không trả thêm text hay markdown. JSON phải hợp lệ và đầy đủ tất cả fields.
        """
        
        safe_prompt = PHIFilter.redact(prompt)
        llm_resp = await self.llm_provider.generate_assessment(safe_prompt)
        
        try:
            data = json.loads(llm_resp.text)
        except Exception as e:
            logger.error(f"Lỗi JSON: {e}")
            data = {"risk": "high", "escalation": "escalate", "suggested_disposition": "human_review", "working_diagnosis": "Lỗi", "rationale": "Fallback an toàn do lỗi parse"}

        return AgentAssessment(
            tier=self.tier, role=self.role,
            risk=RiskLevel(data.get("risk", "high")),
            escalation=EscalationDecision(data.get("escalation", "escalate")),
            differential=[data.get("working_diagnosis", "Unknown")],
            rationale=data.get("rationale", ""),
            suggested_disposition=Disposition(data.get("suggested_disposition", "human_review")),
            token_usage=TokenUsage(
                prompt_tokens=llm_resp.prompt_tokens, completion_tokens=llm_resp.completion_tokens,
                total_tokens=llm_resp.prompt_tokens + llm_resp.completion_tokens
            )
        )