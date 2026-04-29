from fastapi.testclient import TestClient

from tao_medical_ai.interfaces.api.main import app

client = TestClient(app)
API_HEADERS = {"x-api-key": "local-dev-key"}


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_requires_api_key() -> None:
    response = client.post(
        "/v1/cases/analyze",
        json={
            "case_id": "api-001",
            "patient": {"age": 50, "sex": "male"},
            "chief_complaint": "Chest pain",
            "timeline": "1 hour",
            "symptoms": ["chest pain"],
            "vitals": {"spo2": 90, "heart_rate": 100, "systolic_bp": 100},
        },
    )
    assert response.status_code == 401


def test_specialties_endpoint() -> None:
    response = client.get("/v1/specialties", headers=API_HEADERS)
    assert response.status_code == 200
    payload = response.json()
    assert any(item["slug"] == "cardiology" for item in payload)
    assert any(item["slug"] == "neurology" for item in payload)
    assert any(item["slug"] == "general_outpatient" for item in payload)
