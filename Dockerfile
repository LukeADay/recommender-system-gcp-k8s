# Use a base image compatible with Apple M1 (ARM64)
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Install system dependencies for TensorFlow and Google Cloud Storage
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && apt-get clean

# Upgrade pip and install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for Google Cloud Storage
RUN pip install google-auth google-auth-oauthlib google-auth-httplib2

# Copy the scripts into the container
COPY scripts /app/scripts

# Set the default command (optional for testing)
CMD ["python", "/app/scripts/train_ncf.py"]
