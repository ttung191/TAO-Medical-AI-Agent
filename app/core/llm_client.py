import json
import logging
import time
import google.generativeai as genai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config.settings import settings
from app.models.schemas import CostMetrics

logger = logging.getLogger(__name__)

# Hàm kiểm tra xem lỗi có phải do hết Quota (429) không để kích hoạt Retry
def is_rate_limit_error(exception):
    error_msg = str(exception).lower()
    return "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg

class LLMClient:
    def __init__(self):
        # Khởi tạo các API Keys dựa trên file settings
        if settings.GOOGLE_API_KEY:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            self.provider = "gemini"
        elif settings.OPENAI_API_KEY:
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self.provider = "openai"
        else:
            raise ValueError("⚠️ Hệ thống thiếu API Key! Vui lòng kiểm tra lại file .env hoặc cấu hình Streamlit Secrets.")

    # Gắn bùa "Hồi sinh" Tenacity cho văn bản thường
    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(4), 
        wait=wait_exponential(multiplier=5, min=10, max=60), # Đợi từ 10s đến 60s
        reraise=True
    )
    def generate_content(self, prompt: str, system_instruction: str = None) -> str:
        """Hàm lõi gọi LLM trả về văn bản thường, đã được trang bị cơ chế tự động tránh lỗi Spam."""
        
        logger.info(f"⏳ Đang hãm phanh 8 giây để tránh lỗi Quota... Chuẩn bị gọi {self.provider}")
        time.sleep(8)

        try:
            if self.provider == "gemini":
                if system_instruction:
                    model = genai.GenerativeModel(
                        model_name='gemini-2.5-flash',
                        system_instruction=system_instruction
                    )
                else:
                    model = self.model
                    
                response = model.generate_content(prompt)
                return response.text
                
            elif self.provider == "openai":
                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                )
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"❌ LLM gặp sự cố: {str(e)}")
            if is_rate_limit_error(e):
                logger.warning("⚠️ Đụng trần API Free! Tenacity đang kích hoạt đếm lùi để thử lại...")
                raise Exception(f"RateLimitError: {str(e)}") 
            raise e 

    # Gắn bùa "Hồi sinh" Tenacity cho JSON Mode
    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(4), 
        wait=wait_exponential(multiplier=5, min=10, max=60),
        reraise=True
    )
    def generate_json(self, prompt: str, system_instruction: str = None) -> dict:
        """Hàm xuất dữ liệu dạng JSON cho các Agent, có gắn phanh 8 giây."""
        
        logger.info(f"⏳ Đang hãm phanh 8 giây (JSON Mode) để tránh lỗi Quota... Chuẩn bị gọi {self.provider}")
        time.sleep(8)

        try:
            if self.provider == "gemini":
                # Cấu hình ép Gemini trả về chuẩn JSON
                generation_config = {"response_mime_type": "application/json"}
                
                if system_instruction:
                    model = genai.GenerativeModel(
                        model_name='gemini-2.5-flash',
                        system_instruction=system_instruction,
                        generation_config=generation_config
                    )
                else:
                    model = genai.GenerativeModel(
                        model_name='gemini-2.5-flash',
                        generation_config=generation_config
                    )
                    
                response = model.generate_content(prompt)
                return json.loads(response.text)
                
            elif self.provider == "openai":
                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    response_format={ "type": "json_object" } # Ép OpenAI trả JSON
                )
                return json.loads(response.choices[0].message.content)
                
        except Exception as e:
            logger.error(f"❌ Lỗi khi generate JSON: {str(e)}")
            if is_rate_limit_error(e):
                logger.warning("⚠️ Đụng trần API! Tenacity đang kích hoạt đếm lùi...")
                raise Exception(f"RateLimitError: {str(e)}")
            
            # Trả về dict lỗi thay vì sập hệ thống
            return {"error": "Invalid JSON output from LLM", "details": str(e)}
