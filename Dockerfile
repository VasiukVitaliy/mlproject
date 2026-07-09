FROM python:3.13-slim

ENV UV_PROGRESS=off
RUN pip install --no-cache-dir uv

WORKDIR /app

COPY requirements.txt .

RUN uv pip install -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

CMD ["uv", "run", "gunicorn", "--bind", "0.0.0.0:8080", "app:app"]