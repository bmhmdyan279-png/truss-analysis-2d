# syntax=docker/dockerfile:1

# ==========================================
# Stage 1: Builder (Compile and install)
# ==========================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build tools and git (required for setuptools_scm fallback)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy all project files (respects .dockerignore)
# We MUST copy 'src/' so setuptools can find the package
COPY . .

# Create a virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install . scienceplots

# ==========================================
# Stage 2: Runtime (Headless execution)
# ==========================================
FROM python:3.11-slim AS runtime

# Critical for headless matplotlib execution in Docker
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PATH="/opt/venv/bin:$PATH"

# Install runtime system dependencies for Matplotlib & Arabic Reshaper
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libfontconfig1 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the virtual environment and application code from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

# Default command executes the Phase 8 H1 reproducibility test
CMD ["python", "scripts/compute_phase8_h1_test.py"]
