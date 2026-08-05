FROM python:3.10-slim

# Install system dependencies (git is needed for CLIP setup)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (Hugging Face security standard)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app

WORKDIR /app

# Install python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy all project code
COPY --chown=user . .

# Expose port 7860 (Hugging Face Space default port)
EXPOSE 7860

# Launch Streamlit
CMD ["streamlit", "run", "demo/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
