import json
import logging
import time
import google.generativeai as genai
from openai import OpenAI
from config.settings import settings

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        # Khởi tạo nhà cung cấp dựa trên API Key
        if settings.GOOGLE_API_KEY:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.provider = "gemini"
        elif settings.OPENAI_API_KEY:
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self.provider = "openai"
        else:
            raise ValueError("⚠️ Không tìm thấy API Key!")

    def _call_ai(self, is_json=False, *args, **kwargs):
        """Hàm xử lý lõi: Chấp nhận mọi kiểu tham số (args/kwargs) để tránh lỗi 'missing argument'."""
        
        # ⏱️ TỐI ƯU TỐC ĐỘ: Nghỉ 2 giây để né lỗi Quota (429) mà không làm chậm hệ thống quá mức.
        time.sleep(2)

        # 🔍 Tự động nhặt Prompt: Ưu tiên args[0] hoặc từ khóa 'prompt'/'content'
        prompt = args[0] if args else (kwargs.get('prompt') or kwargs.get('content') or "")
        
        # 🔍 Tự động nhặt System Instruction: Chấp nhận cả 'system_instruction' và 'system_prompt'
        system_instr = kwargs.get('system_instruction') or kwargs.get('system_prompt')
        
        # 🔍 Tự động nhặt Model
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
            logger.error(f"❌ LLM Error: {str(e)}")
            # Nếu là lỗi JSON Mode, trả về cấu trúc tối thiểu để Orchestrator không sập
            if is_json:
                return {"diagnosis": "Error", "treatment_plan": f"Sự cố hệ thống: {str(e)}", "escalation_decision": "REJECT"}
            return f"Error: {str(e)}"

    def generate_content(self, *args, **kwargs):
        """Hàm trả về text: Chấp nhận MỌI kiểu tham số đầu vào."""
        return self._call_ai(False, *args, **kwargs)

    def generate_json(self, *args, **kwargs):
        """Hàm trả về JSON: Chấp nhận MỌI kiểu tham số đầu vào."""
        return self._call_ai(True, *args, **kwargs)
