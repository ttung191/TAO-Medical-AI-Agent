import json
import logging
import time
import google.generativeai as genai
from openai import OpenAI
from config.settings import settings

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        # Khởi tạo API
        if settings.GOOGLE_API_KEY:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.provider = "gemini"
        elif settings.OPENAI_API_KEY:
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self.provider = "openai"
        else:
            raise ValueError("⚠️ Thiếu API Key trong cấu hình!")

    def _smart_get_params(self, *args, **kwargs):
        """Bộ lọc thông minh: Tự động nhặt Prompt và System Instruction dù Agent gọi kiểu gì."""
        # 1. Nhặt Prompt (Ưu tiên vị trí đầu tiên, sau đó đến các từ khóa)
        prompt = args[0] if args else (kwargs.get('prompt') or kwargs.get('content') or "")
        
        # 🛡️ Sửa lỗi 'contents must not be empty': Nếu rỗng, gửi một câu lệnh mặc định
        if not str(prompt).strip():
            prompt = "Analyze the provided medical case and provide a structured response."

        # 2. Nhặt System Instruction (Chấp nhận mọi bí danh)
        sys_instr = kwargs.get('system_instruction') or kwargs.get('system_prompt') or kwargs.get('instruction')

        # 3. Nhặt Model
        model_name = kwargs.get('model') or "gemini-1.5-flash"
        
        return prompt, sys_instr, model_name

    def generate_content(self, *args, **kwargs) -> str:
        """Hàm trả về văn bản - Chống mọi loại lỗi tham số."""
        # Nghỉ 1.5 giây: Vừa đủ nhanh để không phải chờ lâu, vừa đủ an toàn để né lỗi 429
        time.sleep(1.5)
        prompt, sys_instr, model_name = self._smart_get_params(*args, **kwargs)

        try:
            if self.provider == "gemini":
                model_obj = genai.GenerativeModel(model_name=model_name, system_instruction=sys_instr)
                return model_obj.generate_content(prompt).text
            else:
                messages = []
                if sys_instr: messages.append({"role": "system", "content": sys_instr})
                messages.append({"role": "user", "content": prompt})
                res = self.client.chat.completions.create(model="gpt-4o-mini", messages=messages)
                return res.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ LLM Content Error: {e}")
            return f"Error: {str(e)}"

    def generate_json(self, *args, **kwargs) -> dict:
        """Hàm trả về JSON - Được thiết kế để không bao giờ làm sập Orchestrator."""
        time.sleep(1.5)
        prompt, sys_instr, model_name = self._smart_get_params(*args, **kwargs)

        try:
            if self.provider == "gemini":
                model_obj = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=sys_instr,
                    generation_config={"response_mime_type": "application/json"}
                )
                response = model_obj.generate_content(prompt)
                return json.loads(response.text)
            else:
                messages = []
                if sys_instr: messages.append({"role": "system", "content": sys_instr})
                messages.append({"role": "user", "content": prompt})
                res = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    response_format={"type": "json_object"}
                )
                return json.loads(res.choices[0].message.content)
        except Exception as e:
            logger.error(f"❌ LLM JSON Error: {e}")
            # 🛡️ TRẢ VỀ CẤU TRÚC AN TOÀN: Chỉ gồm 2 keys để tránh lỗi 'too many values to unpack'
            # Giúp Recruiter và Orchestrator của bạn vẫn có thể chạy tiếp
            return {
                "diagnosis": "System logic error",
                "decision": "REJECT"
            }
