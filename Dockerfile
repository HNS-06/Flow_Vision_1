# Use official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV API_RELOAD=False

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (needed for some python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Copy the rest of the application code
COPY . .

# Generate data and train models (needed for the app to run)
# We run these during build so the image is self-contained
RUN python scripts/generate_sample_data.py
RUN python ml_pipeline/data_preprocessing.py
RUN python ml_pipeline/leak_detection.py
RUN python ml_pipeline/consumption_forecast.py

# Create database directory if not exists (though scripts might have created it)
RUN mkdir -p data/database

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using Gunicorn with Uvicorn workers
# Adjust workers and threads based on available resources
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "backend.app:app", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "8", "--timeout", "0"]
