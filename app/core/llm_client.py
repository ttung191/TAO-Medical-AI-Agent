import json
import logging
import time
import google.generativeai as genai
from openai import OpenAI
from config.settings import settings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

def is_rate_limit_error(e):
    return "429" in str(e) or "quota" in str(e).lower()

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

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=5, max=15),
        reraise=True
    )
    def _execute_ai(self, prompt, system_instruction=None, model=None, is_json=False, **kwargs):
        """Hàm thực thi lõi, bảo đảm nhận đúng System Instruction."""
        # Nghỉ 1.5s để đảm bảo tốc độ (RPM) an toàn cho gói Free
        time.sleep(1.5)

        # Xử lý các bí danh: Nếu Agent truyền 'system_prompt' thay vì 'system_instruction'
        sys_instr = system_instruction or kwargs.get('system_prompt')
        model_name = model or kwargs.get('model') or "gemini-1.5-flash"

        try:
            if self.provider == "gemini":
                config = {"response_mime_type": "application/json"} if is_json else {}
                model_obj = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=sys_instr,
                    generation_config=config
                )
                response = model_obj.generate_content(prompt)
                return response.text if not is_json else json.loads(response.text)
                
            elif self.provider == "openai":
                messages = []
                if sys_instr:
                    messages.append({"role": "system", "content": sys_instr})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model=model_name if "gpt" in model_name else "gpt-4o-mini",
                    messages=messages,
                    response_format={"type": "json_object"} if is_json else None
                )
                res_text = response.choices[0].message.content
                return res_text if not is_json else json.loads(res_text)
                
        except Exception as e:
            if is_rate_limit_error(e): raise e
            raise e

    def generate_content(self, prompt, system_instruction=None, model=None, **kwargs):
        """Hàm text: Khớp hoàn toàn với mọi Agent."""
        try:
            return self._execute_ai(prompt, system_instruction, model, is_json=False, **kwargs)
        except Exception as e:
            return f"Error: {str(e)}"

    def generate_json(self, prompt, system_instruction=None, model=None, **kwargs):
        """Hàm JSON: Khớp hoàn toàn với mọi Agent."""
        try:
            return self._execute_ai(prompt, system_instruction, model, is_json=True, **kwargs)
        except Exception as e:
            logger.error(f"Lỗi JSON: {e}")
            # Trả về cấu trúc 2 phần tử tối thiểu để không bị lỗi 'unpack'
            return {"diagnosis": "Error", "reasoning": str(e)}
