# Use official slim Python image
FROM python:3.9-slim

# Install system dependencies (specifically Tesseract-OCR)
RUN apt-get update && apt-get install -y tesseract-ocr libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install them, avoiding cache to save image payload size
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Run the FastAPI server on port defined by Render (or fallback to 8000)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
