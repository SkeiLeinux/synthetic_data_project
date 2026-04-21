FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/         ./api/
COPY shared/      ./shared/
COPY data_service/ ./data_service/
COPY config_loader.py ./
COPY configs/     ./configs/

RUN mkdir -p data models logs reporter/reports

EXPOSE 8000

CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
