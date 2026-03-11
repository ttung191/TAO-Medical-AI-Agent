import json
import logging
import time
import google.generativeai as genai
from openai import OpenAI
from config.settings import settings

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        if settings.GOOGLE_API_KEY:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.provider = "gemini"
        elif settings.OPENAI_API_KEY:
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self.provider = "openai"
        else:
            raise ValueError("⚠️ Cần API Key để hoạt động!")

    def generate_content(self, prompt, system_instruction=None, **kwargs):
        """Hàm trả về text: Chấp nhận mọi tham số để tránh lỗi 'unexpected argument'."""
        time.sleep(2) # Nghỉ 2s né lỗi 429
        sys_instr = system_instruction or kwargs.get('system_prompt')
        model_name = kwargs.get('model') or "gemini-1.5-flash"
        
        try:
            if self.provider == "gemini":
                model_obj = genai.GenerativeModel(model_name=model_name, system_instruction=sys_instr)
                return model_obj.generate_content(prompt).text
            else:
                messages = []
                if sys_instr: messages.append({"role": "system", "content": sys_instr})
                messages.append({"role": "user", "content": prompt})
                res = self.client.chat.completions.create(model="gpt-4o-mini", messages=messages)
                return res.choices[0].message.content
        except Exception as e:
            logger.error(f"Error: {e}")
            return f"Error: {str(e)}"

    def generate_json(self, prompt, system_instruction=None, **kwargs):
        """Hàm trả về JSON: Bảo đảm trả về đủ 7 trường dữ liệu để không bị lỗi unpack."""
        time.sleep(2) # Nghỉ 2s né lỗi 429
        sys_instr = system_instruction or kwargs.get('system_prompt')
        model_name = kwargs.get('model') or "gemini-1.5-flash"

        try:
            if self.provider == "gemini":
                model_obj = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=sys_instr,
                    generation_config={"response_mime_type": "application/json"}
                )
                response = model_obj.generate_content(prompt)
                return json.loads(response.text)
            else:
                messages = []
                if sys_instr: messages.append({"role": "system", "content": sys_instr})
                messages.append({"role": "user", "content": prompt})
                res = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    response_format={"type": "json_object"}
                )
                return json.loads(res.choices[0].message.content)
        except Exception as e:
            logger.error(f"JSON Error: {e}")
            # TRẢ VỀ ĐẦY ĐỦ 7 TRƯỜNG ĐỂ TRÁNH LỖI UNPACK (EXPECTED 2, GOT 7)
            return {
                "diagnosis_summary": "Sự cố kỹ thuật khi gọi AI",
                "treatment_plan": "Vui lòng thử lại sau vài giây",
                "confidence_score": 0.0,
                "risk_assessment": "HIGH",
                "reasoning": str(e),
                "escalation_decision": "REJECT",
                "feedback_to_lower_tier": "API rate limit or connection issue"
            }
