FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


COPY frontend/requirements.txt frontend-requirements.txt

RUN pip install --no-cache-dir -r frontend-requirements.txt


COPY . .


EXPOSE 8000
EXPOSE 8501


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]