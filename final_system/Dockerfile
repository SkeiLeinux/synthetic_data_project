FROM python:3.11-slim

WORKDIR /app

# Системные зависимости (нужны для psycopg2 и некоторых ML-библиотек)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Torch устанавливаем отдельно с CPU-индексом —
# GPU-версия весит ~2.5 ГБ и не нужна внутри контейнера без nvidia-docker
RUN pip install --no-cache-dir torch~=2.11.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Остальные зависимости (torch уже установлен, pip его пропустит)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Создаём директории которые будут bind-mount'иться
RUN mkdir -p data models logs reporter/reports

EXPOSE 8000

CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]