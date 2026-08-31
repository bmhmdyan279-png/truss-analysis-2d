# syntax=docker/dockerfile:1

# ==========================================
# Stage 1: Builder (Compile wheels)
# ==========================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definitions first to leverage Docker cache
COPY pyproject.toml README.md ./

# Build wheels
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels . scienceplots

# ==========================================
# Stage 2: Runtime (Headless execution)
# ==========================================
FROM python:3.11-slim AS runtime

# Critical for headless matplotlib execution in Docker
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# Install runtime system dependencies for Matplotlib & Arabic Reshaper
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libfontconfig1 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy wheels from builder and install
COPY --from=builder /app/wheels /app/wheels
COPY . .

RUN pip install --no-cache-dir --no-index --find-links=/app/wheels . scienceplots \
    && rm -rf /app/wheels

# Default command executes the Phase 8 H1 reproducibility test
CMD ["python", "scripts/compute_phase8_h1_test.py"]
