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
        wait=wait_exponential(multiplier=2, min=4, max=10),
        reraise=True
    )
    def _ask_ai(self, is_json=False, *args, **kwargs):
        """Hàm xử lý vạn năng: Chấp nhận mọi kiểu tham số từ Agent."""
        # Nghỉ 1.2s để vừa nhanh vừa không bị Google khóa (RPM limit)
        time.sleep(1.2)

        # 1. Tìm Prompt: ưu tiên args[0], sau đó đến key 'prompt' hoặc 'content'
        prompt = ""
        if args:
            prompt = args[0]
        else:
            prompt = kwargs.get('prompt') or kwargs.get('content') or ""

        # 2. Tìm System Instruction: chấp nhận cả 2 tên gọi phổ biến
        system_instr = kwargs.get('system_instruction') or kwargs.get('system_prompt')
        
        # 3. Chọn Model
        model_name = kwargs.get('model') or "gemini-1.5-flash"

        try:
            if self.provider == "gemini":
                config = {"response_mime_type": "application/json"} if is_json else {}
                model_obj = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_instr,
                    generation_config=config
                )
                response = model_obj.generate_content(prompt)
                return response.text if not is_json else json.loads(response.text)
                
            elif self.provider == "openai":
                messages = []
                if system_instr:
                    messages.append({"role": "system", "content": system_instr})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model=kwargs.get('model') or "gpt-4o-mini",
                    messages=messages,
                    response_format={"type": "json_object"} if is_json else None
                )
                res_text = response.choices[0].message.content
                return res_text if not is_json else json.loads(res_text)
                
        except Exception as e:
            if is_rate_limit_error(e):
                logger.warning("⚠️ Đang đợi vì hết hạn mức (429)...")
                raise e # Để tenacity tự thử lại
            raise e

    def generate_content(self, *args, **kwargs):
        """Hàm trả về text: Chấp nhận MỌI tham số đầu vào."""
        try:
            return self._ask_ai(False, *args, **kwargs)
        except Exception as e:
            return f"Error: {str(e)}"

    def generate_json(self, *args, **kwargs):
        """Hàm trả về JSON: Chấp nhận MỌI tham số đầu vào."""
        try:
            return self._ask_ai(True, *args, **kwargs)
        except Exception as e:
            return {
                "diagnosis": "Lỗi hệ thống",
                "reasoning": str(e),
                "escalation_decision": "REJECT"
            }
