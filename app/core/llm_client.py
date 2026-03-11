import json
import logging
import time
import google.generativeai as genai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config.settings import settings
from app.models.schemas import CostMetrics

logger = logging.getLogger(__name__)

# Tùy chỉnh hàm Retry: Thử lại tối đa 4 lần, thời gian đợi tăng dần (10s, 20s, 40s...)
def is_rate_limit_error(exception):
    """Hàm kiểm tra xem lỗi có phải do hết Quota (429) không để kích hoạt Retry."""
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

    # Gắn bùa "Hồi sinh" Tenacity: Tự động thử lại nếu gặp lỗi Rate Limit
    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(4), 
        wait=wait_exponential(multiplier=5, min=10, max=60), # Đợi từ 10s đến 60s
        reraise=True
    )
    def generate_content(self, prompt: str, system_instruction: str = None) -> str:
        """Hàm lõi gọi LLM, đã được trang bị cơ chế tự động tránh lỗi Spam (429)."""
        
        # 🛑 LỚP KHIÊN SỐ 1: Bắt hệ thống nghỉ ngơi 8 giây trước mỗi lần nhấc máy gọi AI
        # Điều này đảm bảo tốc độ tối đa chỉ là ~7 requests/phút (An toàn cho gói Free của Google)
        logger.info(f"⏳ Đang hãm phanh 8 giây để tránh lỗi Quota... Chuẩn bị gọi {self.provider}")
        time.sleep(8)

        try:
            if self.provider == "gemini":
                # Nếu có System Instruction (Ép AI đóng vai), khởi tạo model cấu hình mới
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
                    model="gpt-4o-mini", # Hoặc model mặc định bạn dùng
                    messages=messages,
                )
                return response.choices[0].message.content
                
        except Exception as e:
            # 🛡️ LỚP KHIÊN SỐ 2: Bắt lỗi. Nếu là lỗi 429, đá sang cho Tenacity xử lý đợi và thử lại.
            logger.error(f"❌ LLM gặp sự cố: {str(e)}")
            if is_rate_limit_error(e):
                logger.warning("⚠️ Đụng trần API Free! Tenacity đang kích hoạt đếm lùi để thử lại...")
                raise Exception(f"RateLimitError: {str(e)}") # Bắn lỗi ra để Tenacity bắt được
            raise e # Nếu là lỗi khác (ví dụ sai API key), thì báo lỗi luôn
