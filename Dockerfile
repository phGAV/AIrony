FROM python:3.11-slim as base
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y curl \
    && pip install --no-cache-dir -r requirements.txt

FROM base as backend
COPY app/main.py app/meme_generator.py ./
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base as frontend
COPY app/app.py .
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
