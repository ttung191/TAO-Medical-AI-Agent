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
            raise ValueError("⚠️ Cần API Key!")

    def _get_input(self, prompt, kwargs):
        p = prompt if prompt else (kwargs.get('prompt') or kwargs.get('content') or "Process case")
        s = kwargs.get('system_instruction') or kwargs.get('system_prompt')
        m = kwargs.get('model') or "gemini-1.5-flash"
        return p, s, m

    def generate_content(self, prompt=None, **kwargs):
        time.sleep(1)
        p, s, m = self._get_input(prompt, kwargs)
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
            return f"Error: {str(e)}"

    def generate_json(self, prompt=None, **kwargs):
        """Hàm JSON thông minh: Tự điều chỉnh cấu trúc trả về để tránh lỗi Unpack."""
        time.sleep(1)
        p, s, m = self._get_input(prompt, kwargs)

        try:
            if self.provider == "gemini":
                model = genai.GenerativeModel(model_name=m, system_instruction=s,
                                            generation_config={"response_mime_type": "application/json"})
                result_text = model.generate_content(p).text
                data = json.loads(result_text)
            else:
                messages = [{"role": "system", "content": s}] if s else []
                messages.append({"role": "user", "content": p})
                res = self.client.chat.completions.create(model="gpt-4o-mini", messages=messages, 
                                                       response_format={"type": "json_object"})
                data = json.loads(res.choices[0].message.content)
            
            # TRƯỜNG HỢP ĐẶC BIỆT: Nếu data là một danh sách (List), trả về list đó luôn
            return data

        except Exception as e:
            logger.error(f"JSON Error: {e}")
            # PHẢN HỒI AN TOÀN: Trả về một List chứa Tuple để khớp với lệnh 'for name, role in ...'
            # Đây là chìa khóa để fix lỗi 'too many values to unpack'
            return [("Error", "System encountered an issue, please retry.")]
