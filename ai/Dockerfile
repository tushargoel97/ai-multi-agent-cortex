FROM python:3.12-slim

# Build deps for llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch deps (we don't actually need torch but llama-cpp wheels are CPU)
COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY app ./app

RUN mkdir -p /models

EXPOSE 8100

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]
