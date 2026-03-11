import json
import logging
import time
import google.generativeai as genai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config.settings import settings

logger = logging.getLogger(__name__)

def is_rate_limit_error(exception):
    return "429" in str(exception) or "quota" in str(exception).lower()

class LLMClient:
    def __init__(self):
        if settings.GOOGLE_API_KEY:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.provider = "gemini"
        elif settings.OPENAI_API_KEY:
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self.provider = "openai"
        else:
            raise ValueError("⚠️ Missing API Key!")

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=3, max=15),
        reraise=True
    )
    def _execute_call(self, prompt: str, system_instruction: str = None, is_json: bool = False, model: str = None):
        """Hàm thực thi gọi LLM thực tế."""
        # Nghỉ 1.2s để tránh spam RPM, vừa đủ nhanh vừa an toàn
        time.sleep(1.2)
        
        model_name = model if model else 'gemini-2.5-flash'
        
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
                model=model if model else "gpt-4o-mini",
                messages=messages,
                response_format={ "type": "json_object" } if is_json else None
            )
            return response.choices[0].message.content

    def generate_content(self, *args, **kwargs) -> str:
        """Xử lý linh hoạt mọi kiểu gọi hàm để lấy nội dung text."""
        prompt = args[0] if args else kwargs.get('prompt', '')
        system_instruction = kwargs.get('system_instruction')
        model = kwargs.get('model')
        
        try:
            return self._execute_call(prompt, system_instruction, is_json=False, model=model)
        except Exception as e:
            if is_rate_limit_error(e): raise e
            return f"Error: {str(e)}"

    def generate_json(self, *args, **kwargs) -> dict:
        """Xử lý linh hoạt mọi kiểu gọi hàm để lấy nội dung JSON."""
        prompt = args[0] if args else kwargs.get('prompt', '')
        system_instruction = kwargs.get('system_instruction')
        model = kwargs.get('model')
        
        try:
            result = self._execute_call(prompt, system_instruction, is_json=True, model=model)
            return json.loads(result)
        except Exception as e:
            if is_rate_limit_error(e): raise e
            # Nếu lỗi JSON, trả về cấu trúc để Agent không sập
            return {"diagnosis": "Error", "reasoning": str(e), "escalation_decision": "REJECT"}
