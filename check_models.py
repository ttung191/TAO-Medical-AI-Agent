import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Lỗi: Chưa tìm thấy GOOGLE_API_KEY trong file .env")
else:
    genai.configure(api_key=api_key)
    print(f"🔑 Đang kiểm tra với Key: {api_key[:5]}...*****")
    print("\n📋 DANH SÁCH MODEL KHẢ DỤNG:")
    
    try:
        found_any = False
        for m in genai.list_models():
            # Chỉ lấy các model chat (generateContent)
            if 'generateContent' in m.supported_generation_methods:
                print(f"   ✅ {m.name}")  # Ví dụ: models/gemini-1.5-flash
                found_any = True
        
        if not found_any:
            print("⚠️ Không tìm thấy model nào! Có thể Key bị lỗi hoặc chưa kích hoạt.")
            
    except Exception as e:
        print(f"❌ Lỗi kết nối: {e}")