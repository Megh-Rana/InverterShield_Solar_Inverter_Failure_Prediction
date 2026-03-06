FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY src/ src/
COPY models/ models/
COPY data/processed/ data/processed/
COPY run_pipeline.py .

# Environment
ENV MODEL_DIR=/app/models
ENV DATA_DIR=/app/data/processed
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8000 8501

# Default: run API server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
