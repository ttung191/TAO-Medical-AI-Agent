FROM python:3.11-slim

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -e .
EXPOSE 8000
CMD ["uvicorn", "tao_medical_ai.interfaces.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
