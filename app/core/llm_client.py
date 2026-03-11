import json
import logging
import google.generativeai as genai
from openai import OpenAI
from config.settings import settings

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        # Khởi tạo provider dựa trên API Key trong cấu hình
        if settings.GOOGLE_API_KEY:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.provider = "gemini"
        elif settings.OPENAI_API_KEY:
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self.provider = "openai"
        else:
            raise ValueError("No API key found in settings. Please check your .env or Secrets.")

    def generate_content(self, prompt, system_instruction=None, model=None):
        """Hàm trả về văn bản thuần túy."""
        model_name = model if model else "gemini-1.5-flash"
        
        try:
            if self.provider == "gemini":
                model_obj = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_instruction
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
                    messages=messages
                )
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in generate_content: {e}")
            return f"Error: {str(e)}"

    def generate_json(self, prompt, system_instruction=None, model=None):
        """Hàm trả về dữ liệu cấu trúc JSON."""
        model_name = model if model else "gemini-1.5-flash"
        
        try:
            if self.provider == "gemini":
                generation_config = {"response_mime_type": "application/json"}
                model_obj = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_instruction,
                    generation_config=generation_config
                )
                response = model_obj.generate_content(prompt)
                return json.loads(response.text)
                
            elif self.provider == "openai":
                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model=model if model else "gpt-4o-mini",
                    messages=messages,
                    response_format={"type": "json_object"}
                )
                return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error in generate_json: {e}")
            # Trả về cấu trúc mặc định để tránh lỗi logic ở các tầng trên
            return {
                "diagnosis": "Error",
                "reasoning": str(e),
                "escalation_decision": "REJECT"
            }
