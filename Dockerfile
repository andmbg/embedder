FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Make clear we need 2 env vars: API_TOKEN
LABEL org.opencontainers.image.title="embedder"
LABEL org.opencontainers.image.description="Custom Sentence Embedder service (requires API_TOKEN at runtime)"
LABEL com.fluesterx.required_env="API_TOKEN"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with compatible PyTorch
RUN pip3 install --no-cache-dir \
    torch

# Install required dependencies
RUN pip3 install --no-cache-dir \
    flask \
    gunicorn \
    python-dotenv \
    pandas \
    transformers \
    sentence-transformers

# Create app directory
WORKDIR /app

# Copy service files
COPY embedder_service.py /app/
COPY src/ /app/src/

# Create log directory
RUN mkdir -p /var/log

# Set default environment variables (can be overridden)
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV FLASK_ENV=production
ENV HOST=0.0.0.0
ENV PORT=19000
ENV WORKERS=1
ENV THREADS=2
ENV TIMEOUT=300
ENV API_TOKEN=""

# Expose port
EXPOSE 19000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Start service
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python3", "embedder_service.py"]
