import json
import logging
import google.generativeai as genai
from openai import OpenAI
from config.settings import settings

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        if settings.GOOGLE_API_KEY:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.provider = "gemini"
        elif settings.OPENAI_API_KEY:
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self.provider = "openai"
        else:
            raise ValueError("No API key found!")

    def generate_content(self, prompt, system_instruction=None, **kwargs):
        # Lọc để lấy system instruction từ bất kỳ tên gọi nào Agent dùng
        sys_instr = system_instruction or kwargs.get('system_prompt')
        model_name = kwargs.get('model') or "gemini-1.5-flash"
        
        # Chặn đứng lỗi "contents must not be empty"
        if not prompt:
            prompt = "No input provided"

        try:
            if self.provider == "gemini":
                model_obj = genai.GenerativeModel(model_name=model_name, system_instruction=sys_instr)
                return model_obj.generate_content(prompt).text
            elif self.provider == "openai":
                messages = []
                if sys_instr: messages.append({"role": "system", "content": sys_instr})
                messages.append({"role": "user", "content": prompt})
                res = self.client.chat.completions.create(model="gpt-4o-mini", messages=messages)
                return res.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return f"Error: {str(e)}"

    def generate_json(self, prompt, system_instruction=None, **kwargs):
        sys_instr = system_instruction or kwargs.get('system_prompt')
        model_name = kwargs.get('model') or "gemini-1.5-flash"
        
        if not prompt:
            prompt = "No input provided for JSON generation"

        try:
            if self.provider == "gemini":
                model_obj = genai.GenerativeModel(
                    model_name=model_name, 
                    system_instruction=sys_instr,
                    generation_config={"response_mime_type": "application/json"}
                )
                return json.loads(model_obj.generate_content(prompt).text)
            elif self.provider == "openai":
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
            logger.error(f"LLM Error: {e}")
            # Trả về đúng 2 phím (keys) để tránh lỗi "too many values to unpack" 
            # nếu orchestrator của bạn đang unpack 2 biến
            return {"name": "Error", "role": str(e)}
