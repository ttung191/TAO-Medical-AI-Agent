import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # --- API Keys ---
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # --- Model Configuration ---
    # CHÚNG TA DÙNG TÊN ĐƠN GIẢN NHẤT
    # Code trong llm_client.py sẽ tự động thêm "models/" nếu cần.
    
    # Tier 1: Tốc độ cao
    TIER_1_MODEL = "gemini-1.5-flash"
    
    # Tier 2: Cân bằng
    TIER_2_MODEL = "gemini-1.5-flash"
    
    # Tier 3: Thông minh (Nếu tài khoản bạn lỗi Pro, hãy dùng Flash luôn cho an toàn)
    TIER_3_MODEL = "gemini-1.5-flash" 

    # --- Logic Thresholds ---
    CONFIDENCE_THRESHOLD_LOW = 0.7  
    CONFIDENCE_THRESHOLD_HIGH = 0.9 

settings = Settings()