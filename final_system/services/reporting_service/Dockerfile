# services/reporting_service/Dockerfile
# Контекст сборки: final_system/
# docker compose build reporting_service

FROM python:3.11-slim

WORKDIR /app

COPY services/reporting_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Общие Pydantic-схемы
COPY shared/ ./shared/
# Код самого сервиса
COPY services/reporting_service/ ./services/reporting_service/

RUN mkdir -p /data/reports

EXPOSE 8004

CMD ["uvicorn", "services.reporting_service.main:app", \
     "--host", "0.0.0.0", "--port", "8004", "--workers", "1"]
