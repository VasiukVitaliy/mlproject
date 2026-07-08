FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install uv && uv pip install --system -r requirements.txt

COPY . .

CMD [ "python", "app.py" ]