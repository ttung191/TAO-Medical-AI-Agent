import json
import logging
import time
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
            raise ValueError("⚠️ API Key missing!")

    def _get_input(self, prompt, kwargs):
        p = prompt if prompt else (kwargs.get('prompt') or kwargs.get('content') or "Process medical case")
        s = kwargs.get('system_instruction') or kwargs.get('system_prompt')
        m = kwargs.get('model') or "gemini-1.5-flash"
        return str(p), s, m

    def generate_content(self, prompt=None, **kwargs):
        time.sleep(1.2)
        p, s, m = self._get_input(prompt, kwargs)
        try:
            if self.provider == "gemini":
                model = genai.GenerativeModel(model_name=m, system_instruction=s)
                return model.generate_content(p).text
            else:
                messages = [{"role": "system", "content": s}] if s else []
                messages.append({"role": "user", "content": p})
                res = self.client.chat.completions.create(model="gpt-4o-mini", messages=messages)
                return res.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"

    def generate_json(self, prompt=None, **kwargs):
        time.sleep(1.2)
        p, s, m = self._get_input(prompt, kwargs)
        try:
            if self.provider == "gemini":
                model = genai.GenerativeModel(model_name=m, system_instruction=s,
                                            generation_config={"response_mime_type": "application/json"})
                return json.loads(model.generate_content(p).text)
            else:
                messages = [{"role": "system", "content": s}] if s else []
                messages.append({"role": "user", "content": p})
                res = self.client.chat.completions.create(model="gpt-4o-mini", messages=messages, 
                                                       response_format={"type": "json_object"})
                return json.loads(res.choices[0].message.content)
        except Exception as e:
            logger.error(f"JSON Error: {e}")
            return {"error": str(e), "status": "failed"}
