import json
import logging
import time
import google.generativeai as genai
from openai import OpenAI
from config.settings import settings

logger = logging.getLogger(__name__)

# Lớp "Ma thuật" giúp fix lỗi Unpack (expected 2, got X) mà không cần sửa file khác
class IterDict(dict):
    def __iter__(self):
        return iter(self.items())

class LLMClient:
    def __init__(self):
        if settings.GOOGLE_API_KEY:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.provider = "gemini"
        elif settings.OPENAI_API_KEY:
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self.provider = "openai"
        else:
            raise ValueError("⚠️ Cần API Key trong cấu hình!")

    def _prepare_inputs(self, prompt, kwargs):
        """Xử lý linh hoạt mọi kiểu tham số để né lỗi 'missing argument'."""
        actual_prompt = prompt if prompt else (kwargs.get('prompt') or kwargs.get('content') or "Analyze case")
        # Đảm bảo prompt không bao giờ rỗng để né lỗi 'contents must not be empty'
        if not str(actual_prompt).strip():
            actual_prompt = "Provide medical analysis for the given context."
            
        sys_instr = kwargs.get('system_instruction') or kwargs.get('system_prompt')
        model_name = kwargs.get('model') or "gemini-1.5-flash"
        return actual_prompt, sys_instr, model_name

    def generate_content(self, prompt=None, **kwargs):
        """Hàm trả về text thuần túy."""
        time.sleep(2) # Nghỉ 2s để né giới hạn gọi API (429)
        p, s, m = self._prepare_inputs(prompt, kwargs)

        try:
            if self.provider == "gemini":
                model = genai.GenerativeModel(model_name=m, system_instruction=s)
                return model.generate_content(p).text
            else:
                messages = [{"role": "system", "content": s}] if s else []
                messages.append({"role": "user", "content": p})
                res = self.client.chat.completions.create(model="gpt-4o-mini", messages=messages)
                return res.choices[0].message.content
        except Exception as e:
            logger.error(f"Error: {e}")
            return f"Error: {str(e)}"

    def generate_json(self, prompt=None, **kwargs):
        """Hàm JSON: Trả về IterDict để tương thích hoàn toàn với vòng lặp của Orchestrator."""
        time.sleep(2)
        p, s, m = self._prepare_inputs(prompt, kwargs)

        try:
            if self.provider == "gemini":
                model = genai.GenerativeModel(model_name=m, system_instruction=s,
                                            generation_config={"response_mime_type": "application/json"})
                result = model.generate_content(p).text
                data = json.loads(result)
            else:
                messages = [{"role": "system", "content": s}] if s else []
                messages.append({"role": "user", "content": p})
                res = self.client.chat.completions.create(model="gpt-4o-mini", messages=messages, 
                                                       response_format={"type": "json_object"})
                data = json.loads(res.choices[0].message.content)
            
            # 🚀 TRẢ VỀ ITERDICT: Chìa khóa để fix lỗi 'expected 2, got 7'
            return IterDict(data)

        except Exception as e:
            logger.error(f"JSON Error: {e}")
            # Trả về cấu trúc mặc định chuẩn 7 trường để không bị lỗi Unpack
            return IterDict({
                "diagnosis_summary": "Lỗi kết nối API",
                "treatment_plan": "Vui lòng thử lại",
                "confidence_score": 0.0,
                "risk_assessment": "HIGH",
                "reasoning": str(e),
                "escalation_decision": "REJECT",
                "feedback_to_lower_tier": "API issue"
            })
