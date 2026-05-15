# services/synthesis_service/Dockerfile
# Контекст сборки: final_system/
# docker compose build synthesis_service

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY services/synthesis_service/requirements.txt .

# Torch с CUDA 12.8 (требует NVIDIA Container Toolkit на хосте)
RUN pip install --no-cache-dir torch~=2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Остальные зависимости (torch уже установлен, pip его пропустит)
RUN pip install --no-cache-dir -r requirements.txt

# Генераторы и их зависимости
COPY synthesizer/ ./synthesizer/
# config_loader нужен только для валидации GeneratorYamlConfig (lazy-импорты
# evaluator/reporter внутри .to_*_config() в synthesis_service не вызываются)
COPY config_loader.py .
# Общие Pydantic-схемы
COPY shared/ ./shared/
# Код самого сервиса
COPY services/synthesis_service/ ./services/synthesis_service/

RUN mkdir -p /data/synth /data/models

EXPOSE 8002

CMD ["uvicorn", "services.synthesis_service.main:app", \
     "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]
