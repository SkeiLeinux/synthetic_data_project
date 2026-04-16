# services/data_service/Dockerfile
# Контекст сборки: final_system/
# docker compose build data_service

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY services/data_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Исходный код монолитного модуля data_service (DataProcessor, DataIO)
COPY data_service/ ./data_service/
# logger_config нужен data_service/processor.py
COPY logger_config.py .
# Общие Pydantic-схемы
COPY shared/ ./shared/
# Код самого сервиса (main.py, router.py, settings.py)
COPY services/data_service/ ./services/data_service/

RUN mkdir -p /data/datasets /data/splits

EXPOSE 8001

CMD ["uvicorn", "services.data_service.main:app", \
     "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
