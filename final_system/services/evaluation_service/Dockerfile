# services/evaluation_service/Dockerfile
# Контекст сборки: final_system/
# docker compose build evaluation_service

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY services/evaluation_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Логика оценщиков (без torch/opacus/sdv)
COPY evaluator/ ./evaluator/
# Общие Pydantic-схемы
COPY shared/ ./shared/
# Код самого сервиса
COPY services/evaluation_service/ ./services/evaluation_service/

EXPOSE 8003

CMD ["uvicorn", "services.evaluation_service.main:app", \
     "--host", "0.0.0.0", "--port", "8003", "--workers", "1"]
