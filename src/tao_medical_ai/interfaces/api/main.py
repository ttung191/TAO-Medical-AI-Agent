from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tao_medical_ai.contracts.case import StructuredCase
from tao_medical_ai.orchestration.tao_orchestrator import TAOOrchestrator

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="TAO Enterprise API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

orchestrator = TAOOrchestrator()

@app.post("/v2/cases/analyze")
async def analyze_case(case: StructuredCase):
    return await orchestrator.process_case(case)

@app.get("/health")
def health():
    return {"status": "ok"}