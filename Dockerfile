FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt README.md /app/
COPY src /app/src
COPY config /app/config
COPY scripts /app/scripts

RUN pip install --no-cache-dir -e .

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "smartcity_vision.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
