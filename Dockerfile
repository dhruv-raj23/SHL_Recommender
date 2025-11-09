# Use a stable Python base
FROM python:3.10-slim

# Set working dir
WORKDIR /app

# System deps required by some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl wget unzip libgomp1 libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install Python deps
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip --no-cache-dir install -r /app/requirements.txt

# Copy project files
COPY . /app

# Give permission to start script
RUN chmod +x /app/start.sh

# Expose Gradio default port
EXPOSE 7860

# Start
CMD ["/app/start.sh"]
