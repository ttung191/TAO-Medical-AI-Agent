from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, JSON, Boolean, Float
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class ClinicalCase(Base):
    """Lưu trữ thông tin ca bệnh đầu vào, thiết kế chuẩn hóa cho luồng bệnh án"""
    __tablename__ = 'clinical_cases'

    id = Column(String(50), primary_key=True) # Dùng UUID
    patient_age = Column(Integer)
    patient_sex = Column(String(10))
    chief_complaint = Column(String(500))
    symptoms = Column(JSON) # List các triệu chứng
    medical_history = Column(JSON) # Tiền sử bệnh lý, ICD-10 codes nếu có
    current_medications = Column(JSON) # Thuốc đang sử dụng (Rất quan trọng để check tương tác thuốc)
    
    # Metadata quản lý
    status = Column(String(20), default="processing") # processing, human_review, completed
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    agent_outputs = relationship("AgentOutput", back_populates="case")

class AgentOutput(Base):
    """Lưu vết (Audit Trail) TỪNG quyết định của TỪNG Agent"""
    __tablename__ = 'agent_outputs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String(50), ForeignKey('clinical_cases.id'))
    
    agent_role = Column(String(50)) # VD: triage_nurse, cardiologist
    tier = Column(Integer)
    
    # Quyết định lâm sàng
    working_diagnosis = Column(String(200))
    risk_level = Column(String(20))
    confidence_score = Column(Float)
    suggested_disposition = Column(String(50))
    
    # Raw data & LLM Tracing
    rationale = Column(String) # Lý luận của Agent
    llm_provider = Column(String(50)) # VD: gemini-2.5-flash, gpt-4o
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    case = relationship("ClinicalCase", back_populates="agent_outputs")