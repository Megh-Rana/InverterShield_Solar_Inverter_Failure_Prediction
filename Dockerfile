FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY src/ src/
COPY models/ models/

# Environment
ENV MODEL_DIR=/app/models
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
