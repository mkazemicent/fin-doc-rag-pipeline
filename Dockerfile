# Use an official lightweight Python image
FROM python:3.12-slim as base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies required for build and runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data /app/data/raw /app/data/processed /app/data/chroma_db && \
    chown -R appuser:appuser /app

# --- METADATA & DEPENDENCY LAYER ---
# Copy ONLY the files needed to resolve dependencies and build metadata
COPY pyproject.toml README.md ./
# We also need to copy the directory structure for hatchling to validate the wheel targets
COPY src/ ./src/
COPY app/ ./app/

# Install dependencies and the package itself
RUN pip install --upgrade pip && \
    pip install .

# Download required SpaCy model for Microsoft Presidio
RUN python -m spacy download en_core_web_lg

# Switch to non-root user
USER appuser

# Copy the rest of the application code
COPY --chown=appuser:appuser . .

# Expose Streamlit's default port
EXPOSE 8501

# Healthcheck to ensure the UI is responsive
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Start the application
ENTRYPOINT ["streamlit", "run", "app/main.py", "--server.address=0.0.0.0"]
