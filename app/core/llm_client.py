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
            raise ValueError("⚠️ Không tìm thấy API Key!")

    def _prepare_params(self, kwargs):
        """Hàm phụ để xử lý sự lệch pha giữa các Agent."""
        # Chấp nhận cả 'system_prompt' và 'system_instruction'
        system_instruction = kwargs.get('system_instruction') or kwargs.get('system_prompt')
        model_name = kwargs.get('model') or "gemini-1.5-flash"
        return system_instruction, model_name

    def generate_content(self, prompt, **kwargs):
        """Hàm text: Chấp nhận mọi tham số dư thừa để không bị sập."""
        time.sleep(1.5) # Nghỉ ngắn để né lỗi 429
        system_instruction, model_name = self._prepare_params(kwargs)
        
        try:
            if self.provider == "gemini":
                model_obj = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_instruction
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
                    messages=messages
                )
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Lỗi generate_content: {e}")
            return f"Error: {str(e)}"

    def generate_json(self, prompt, **kwargs):
        """Hàm JSON: Chấp nhận mọi tham số dư thừa để không bị sập."""
        time.sleep(1.5) # Nghỉ ngắn để né lỗi 429
        system_instruction, model_name = self._prepare_params(kwargs)
        
        try:
            if self.provider == "gemini":
                generation_config = {"response_mime_type": "application/json"}
                model_obj = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_instruction,
                    generation_config=generation_config
                )
                response = model_obj.generate_content(prompt)
                return json.loads(response.text)
                
            elif self.provider == "openai":
                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model=kwargs.get('model') or "gpt-4o-mini",
                    messages=messages,
                    response_format={"type": "json_object"}
                )
                return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Lỗi generate_json: {e}")
            return {
                "diagnosis": "Lỗi xử lý",
                "reasoning": str(e),
                "escalation_decision": "REJECT"
            }
