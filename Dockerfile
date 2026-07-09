FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install uv && uv pip install --system -r requirements.txt

COPY . .

CMD ["uv", "run", "--with", "gunicorn", "gunicorn", "--bind", "0.0.0.0:8000", "app:app"]