import json
import logging
import google.generativeai as genai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from config.settings import settings
from app.models.schemas import CostMetrics

if settings.GOOGLE_API_KEY:
    genai.configure(api_key=settings.GOOGLE_API_KEY)

class LLMClient:
    def __init__(self):
        self.logger = logging.getLogger("LLMClient")
        self.openai_client = None
        
        # Cấu hình cứng model Gemini 2.5 Flash
        self.default_google_model = "models/gemini-2.5-flash"

        if settings.OPENAI_API_KEY:
            self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def _calculate_cost(self, model: str, input_tok: int, output_tok: int) -> float:
        # Giá Flash (ước tính)
        price = {"input": 0.075, "output": 0.30}
        cost = (input_tok * price["input"] / 1_000_000) + (output_tok * price["output"] / 1_000_000)
        return round(cost, 8)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_json(self, model: str, system_prompt: str, user_prompt: str):
        try:
            # === GOOGLE GEMINI ===
            if "gemini" in model.lower() or "flash" in model.lower():
                target_model = self.default_google_model
                
                model_instance = genai.GenerativeModel(
                    model_name=target_model,
                    generation_config={"response_mime_type": "application/json"}
                )
                
                combined_prompt = f"System Instruction: {system_prompt}\n\nUser Input: {user_prompt}"
                response = model_instance.generate_content(combined_prompt)
                
                content_text = response.text
                usage = response.usage_metadata
                
                in_tok = usage.prompt_token_count
                out_tok = usage.candidates_token_count
                cost = self._calculate_cost(target_model, in_tok, out_tok)
                
                return json.loads(content_text), CostMetrics(
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    total_cost_usd=cost,
                    model_name=target_model
                )

            # === OPENAI ===
            elif self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                content = response.choices[0].message.content
                usage = response.usage
                
                cost = self._calculate_cost(model, usage.prompt_tokens, usage.completion_tokens)
                
                return json.loads(content), CostMetrics(
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    total_cost_usd=cost,
                    model_name=model
                )
            
            else:
                raise Exception("Chưa cấu hình API Key!")

        except Exception as e:
            self.logger.error(f"Lỗi LLM: {str(e)}")
            return {
                "diagnosis": "Error in processing",
                "treatment": "System Error",
                "risk_level": "MODERATE",
                "confidence": 0.0,
                "reasoning": f"LLM Error: {str(e)}"
            }, CostMetrics(model_name="error")