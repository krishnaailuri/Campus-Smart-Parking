# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set environment variables to prevent python from buffering stdout/stderr and suppress interactive prompts
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for building some Python packages and basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy requirements file to working directory
COPY requirements.txt .

# Upgrade pip and install Python dependencies from requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy all application files into the container
COPY . .

# Expose the port Streamlit will run on
EXPOSE 8501

# Run the Streamlit app by default with configuration for CORS disabled, on port 8501, and headless mode
CMD ["streamlit", "run", "app.py", "--server.enableCORS=false", "--server.port=8501", "--server.headless=true"]
