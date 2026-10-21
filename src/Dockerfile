ARG PY_VERSION=3.9-slim

FROM python:${PY_VERSION}

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "api_model:app", "--host", "0.0.0.0", "--port", "8000"]