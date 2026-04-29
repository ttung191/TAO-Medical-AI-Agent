install:
	pip install -e .[dev]

test:
	pytest

lint:
	ruff check src tests evaluation

run-api:
	uvicorn tao_medical_ai.interfaces.api.main:app --reload --port 8000

run-ui:
	streamlit run src/tao_medical_ai/interfaces/streamlit/app.py

eval:
	python evaluation/run_eval.py
