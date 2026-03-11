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
            raise ValueError("⚠️ Thiếu API Key!")

    def _call_ai(self, is_json=False, **kwargs):
        """Hàm xử lý lõi, linh hoạt với mọi loại tham số truyền vào."""
        # Nghỉ 1.5s để đảm bảo không bị Google chặn vì gọi quá nhanh
        time.sleep(1.5)
        
        # Nhặt nhạnh tham số từ túi kwargs bất kể tên gọi là gì
        prompt = kwargs.get('prompt') or kwargs.get('content') or ""
        system_instruction = kwargs.get('system_instruction') or kwargs.get('system_prompt')
        model_name = kwargs.get('model') or "gemini-1.5-flash"
        
        if self.provider == "gemini":
            config = {"response_mime_type": "application/json"} if is_json else {}
            model_obj = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_instruction,
                generation_config=config
            )
            response = model_obj.generate_content(prompt)
            return response.text
            
        elif self.provider == "openai":
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=kwargs.get('model') or "gpt-4o-mini",
                messages=messages,
                response_format={"type": "json_object"} if is_json else None
            )
            return response.choices[0].message.content

    def generate_content(self, prompt=None, **kwargs):
        """Hàm text: prompt bây giờ là tùy chọn, không sợ bị thiếu tham số."""
        if prompt: kwargs['prompt'] = prompt
        try:
            return self._call_ai(is_json=False, **kwargs)
        except Exception as e:
            logger.error(f"Lỗi Content: {e}")
            return f"Error: {str(e)}"

    def generate_json(self, prompt=None, **kwargs):
        """Hàm JSON: prompt bây giờ là tùy chọn, chấp nhận mọi kiểu gọi."""
        if prompt: kwargs['prompt'] = prompt
        try:
            res = self._call_ai(is_json=True, **kwargs)
            return json.loads(res)
        except Exception as e:
            logger.error(f"Lỗi JSON: {e}")
            return {
                "diagnosis": "Lỗi xử lý hệ thống",
                "reasoning": f"Sự cố API hoặc JSON format: {str(e)}",
                "escalation_decision": "REJECT"
            }
