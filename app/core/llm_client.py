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
            raise ValueError("⚠️ Cần API Key trong Secrets!")

    def _get_actual_prompt(self, prompt, kwargs):
        """Hàm nhặt nhạnh Prompt từ mọi ngóc ngách."""
        if prompt: return prompt
        return kwargs.get('prompt') or kwargs.get('content') or "Please analyze the case."

    def generate_content(self, prompt=None, **kwargs):
        """Hàm text: Không bao giờ báo lỗi thiếu tham số."""
        time.sleep(1) # Phanh nhẹ 1 giây để né giới hạn Google
        
        actual_prompt = self._get_actual_prompt(prompt, kwargs)
        sys_instr = kwargs.get('system_instruction') or kwargs.get('system_prompt')
        model_name = kwargs.get('model') or "gemini-1.5-flash"

        try:
            if self.provider == "gemini":
                model_obj = genai.GenerativeModel(model_name=model_name, system_instruction=sys_instr)
                return model_obj.generate_content(actual_prompt).text
            else:
                messages = []
                if sys_instr: messages.append({"role": "system", "content": sys_instr})
                messages.append({"role": "user", "content": actual_prompt})
                res = self.client.chat.completions.create(model="gpt-4o-mini", messages=messages)
                return res.choices[0].message.content
        except Exception as e:
            logger.error(f"Error: {e}")
            return f"Error: {str(e)}"

    def generate_json(self, prompt=None, **kwargs):
        """Hàm JSON: Trả về đầy đủ cấu trúc để Orchestrator không bị sập."""
        time.sleep(1) 
        
        actual_prompt = self._get_actual_prompt(prompt, kwargs)
        sys_instr = kwargs.get('system_instruction') or kwargs.get('system_prompt')
        model_name = kwargs.get('model') or "gemini-1.5-flash"

        try:
            if self.provider == "gemini":
                model_obj = genai.GenerativeModel(
                    model_name=model_name, 
                    system_instruction=sys_instr,
                    generation_config={"response_mime_type": "application/json"}
                )
                response = model_obj.generate_content(actual_prompt)
                return json.loads(response.text)
            else:
                messages = []
                if sys_instr: messages.append({"role": "system", "content": sys_instr})
                messages.append({"role": "user", "content": actual_prompt})
                res = self.client.chat.completions.create(
                    model="gpt-4o-mini", messages=messages, 
                    response_format={"type": "json_object"}
                )
                return json.loads(res.choices[0].message.content)
        except Exception as e:
            logger.error(f"JSON Error: {e}")
            # TRẢ VỀ ĐỦ CÁC TRƯỜNG ĐỂ TRÁNH LỖI 'UNPACK' TRONG ORCHESTRATOR
            return {
                "diagnosis_summary": "System Logic Error",
                "treatment_plan": "Retry required",
                "confidence_score": 0.0,
                "risk_assessment": "HIGH",
                "reasoning": str(e),
                "escalation_decision": "REJECT",
                "feedback_to_lower_tier": "API issue"
            }
