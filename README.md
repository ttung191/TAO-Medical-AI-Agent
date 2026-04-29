# TAO Medical AI Agent v3

Đây là bản **multi-specialty refactor** của project, giữ đúng tinh thần **TAO + 3 agent** nhưng chuyển hẳn sang hướng một sản phẩm có **3 service line rõ ràng** thay vì một scaffold triage chung:

- **Cardiology Triage**
- **Neurology Stroke Triage**
- **General Outpatient Assistant**

## Mục tiêu của bản v3

Bản v2 đã có khung sản phẩm. Bản v3 đi thêm một bước quan trọng: biến hệ thống từ một triage engine tổng quát thành một nền tảng có **pathway theo specialty**, có thể mở rộng như một product thực tế.

Các thay đổi chính:

- thêm **specialty profiles** và **service line** rõ ràng
- routing theo **auto-detect** hoặc **preferred_domain**
- mỗi specialty có **intake checklist**, **handoff questions**, **evidence pack**, **next-step policy** riêng
- sửa logic để hỗ trợ tốt hơn cho:
  - đau ngực / hồi hộp / ngất nghi tim mạch
  - FAST-positive / focal deficit nghi đột quỵ
  - ca ngoại trú nguy cơ thấp đến trung bình
- thêm endpoint:
  - `GET /v1/specialties`
  - `GET /v1/intake-templates/{domain}`
- nâng cấp Streamlit UI với preset theo specialty
- thêm evaluation set mới cho cả 3 service line
- thêm negation-aware keyword matching để tránh lỗi kiểu `no syncope` nhưng vẫn bị bắt là red flag

## 3 agent vẫn được giữ nguyên

- **Tier 1 Nurse**
  - intake
  - red flags
  - data completeness
- **Tier 2 GP**
  - differential
  - pathway reasoning
  - escalation decision
- **Tier 3 Specialist**
  - cardiologist / stroke neurologist / outpatient supervisor
  - conservative override
  - human review gate cho ca high-risk

## Kiến trúc

```text
Operator / API Client
  -> FastAPI / Streamlit
  -> DomainRouter
  -> Tier1 Nurse
  -> Tier2 GP
  -> Tier3 Specialist (nếu cần)
  -> EvidenceRetriever
  -> PolicyEngine
  -> AuditLogger / HumanReviewQueue
```

## Supported specialties

### 1) Cardiology Triage
Dùng cho:
- chest pain
- chest pressure
- palpitations
- syncope / near syncope
- exertional dyspnea

Đặc điểm:
- red flags cho ACS / arrhythmia
- nhấn mạnh hypotension, hypoxemia, diaphoresis, exertional onset
- giữ disposition bảo thủ hơn cho ca tim mạch

### 2) Neurology Stroke Triage
Dùng cho:
- facial droop
- speech difficulty / slurred speech
- unilateral weakness / numbness
- sudden severe headache
- seizure-like / confusion patterns

Đặc điểm:
- bắt last-known-well
- escalates sớm khi có FAST-like pattern
- chuyên gia tầng 3 là stroke neurologist

### 3) General Outpatient Assistant
Dùng cho:
- sore throat
- cough
- viral-like symptoms
- GI symptoms mức nhẹ đến trung bình
- non-emergent ambulatory support

Đặc điểm:
- ưu tiên self-care / primary-care khi an toàn
- không cho self-care khi thiếu dữ liệu quan trọng
- giữ được separation với hai pathway high-consequence

## Chạy local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
cp .env.example .env
pytest
uvicorn tao_medical_ai.interfaces.api.main:app --reload --port 8000
```

## Streamlit

```bash
streamlit run src/tao_medical_ai/interfaces/streamlit/app.py
```

## API

Header bắt buộc:

```text
x-api-key: local-dev-key
```

Endpoints:

- `GET /health`
- `GET /ready`
- `GET /v1/specialties`
- `GET /v1/intake-templates/{domain}`
- `POST /v1/cases/analyze`

## Ví dụ request

```json
{
  "case_id": "cardio-demo-001",
  "preferred_domain": "cardiology",
  "patient": {
    "age": 58,
    "sex": "male"
  },
  "chief_complaint": "Chest pain and shortness of breath",
  "timeline": "Started 2 hours ago while walking upstairs",
  "symptoms": ["chest pain", "dyspnea", "diaphoresis"],
  "vitals": {
    "spo2": 89,
    "heart_rate": 118,
    "systolic_bp": 92,
    "temperature_c": 37.0
  },
  "history": ["hypertension", "type 2 diabetes"],
  "medications": ["amlodipine"],
  "allergies": [],
  "notes": "Pressure-like pain radiating to left arm."
}
```

## Evaluation

```bash
PYTHONPATH=src python evaluation/run_eval.py
```

Current built-in deterministic eval:
- **Risk match accuracy: 100%**
- **Domain match accuracy: 100%**
- trên bộ case mẫu trong `evaluation/gold_cases.jsonl`

## Cảnh báo quan trọng

Đây vẫn là **clinical decision support scaffold**, chưa phải medical product sẵn sàng deploy cho bệnh nhân thật. Trước khi mang vào môi trường thật, vẫn cần:

- dữ liệu thật theo từng specialty
- benchmark ngoài bộ synthetic/demo
- human review workflow có DB/queue thật
- authn/authz mạnh hơn, RBAC, SSO
- monitoring, audit, incident response
- privacy + legal + clinical governance review
- validation prospective

## Production roadmap tiếp theo

- Postgres cho audit + review queue
- Redis cho rate limit
- OpenTelemetry / Prometheus
- guideline store có versioning
- policy packs theo từng specialty sâu hơn
- doctor feedback loop / error analysis dashboard
- clinician-facing review console
