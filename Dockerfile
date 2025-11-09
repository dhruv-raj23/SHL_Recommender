# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /code

# Copy everything
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Hugging Face port
EXPOSE 7860

# Run Gradio app (not FastAPI)
CMD ["python", "app.py"]
