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
        wait=wait_exponential(multiplier=2, min=5, max=30),
        reraise=True
    )
    def _call_llm(self, is_json: bool = False, **kwargs):
        """Hàm xử lý linh hoạt mọi tham số truyền vào."""
        # Chỉ nghỉ tối thiểu 1.5s để giữ an toàn cho RPM (Requests Per Minute)
        time.sleep(1.5) 
        
        # Lấy prompt từ bất kỳ đâu (vị trí đầu tiên hoặc từ khóa 'prompt')
        prompt = kwargs.get('prompt') or ""
        system_instruction = kwargs.get('system_instruction')
        model_name = kwargs.get('model') or 'gemini-2.5-flash'
        
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
                response_format={ "type": "json_object" } if is_json else None
            )
            return response.choices[0].message.content

    # Sử dụng *args và **kwargs để chấp nhận MỌI kiểu gọi từ Agent
    def generate_content(self, *args, **kwargs) -> str:
        if args: kwargs['prompt'] = args[0]
        try:
            return self._call_llm(is_json=False, **kwargs)
        except Exception as e:
            if is_rate_limit_error(e): raise e
            return f"Error: {str(e)}"

    def generate_json(self, *args, **kwargs) -> dict:
        if args: kwargs['prompt'] = args[0]
        try:
            result = self._call_llm(is_json=True, **kwargs)
            return json.loads(result)
        except Exception as e:
            if is_rate_limit_error(e): raise e
            return {"error": "Invalid JSON", "details": str(e)}
