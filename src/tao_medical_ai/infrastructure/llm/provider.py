import os
import google.generativeai as genai
from abc import ABC, abstractmethod
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class LLMResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int

class LLMProvider(ABC):
    @abstractmethod
    async def generate_assessment(self, prompt: str) -> LLMResponse: pass

class GeminiProvider(LLMProvider):
    def __init__(self, model_name="gemini-2.5-flash"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_assessment(self, prompt: str) -> LLMResponse:
        try:
            res = await self.model.generate_content_async(prompt, generation_config=self.generation_config)
            p_tokens = res.usage_metadata.prompt_token_count if res.usage_metadata else 0
            c_tokens = res.usage_metadata.candidates_token_count if res.usage_metadata else 0
            return LLMResponse(text=res.text, prompt_tokens=p_tokens, completion_tokens=c_tokens)
        except Exception as e:
            logger.error(f"Lỗi API Gemini: {str(e)}")
            raise e

class LLMFactory:
    @staticmethod
    def get_provider(tier: int) -> LLMProvider:
        return GeminiProvider("gemini-2.5-flash") if tier == 1 else GeminiProvider("gemini-2.5-flash")