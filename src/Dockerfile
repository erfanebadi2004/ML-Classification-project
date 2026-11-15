ARG PY_VERSION=3.12-slim

FROM python:${PY_VERSION}

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

RUN for i in 1 2 3; do \
        apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* && break || \
        sleep 3; \
    done

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api_model:app", "--host", "0.0.0.0", "--port", "8000"]