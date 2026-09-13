# syntax=docker/dockerfile:1

# ==========================================
# Stage 1: Builder (compile and install)
# ==========================================
# Base image pinned by tag AND digest for reproducible builds.
FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534 AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build toolchain (git is required by setuptools_scm at build time)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy all project files (respects .dockerignore)
COPY . .

# Create a virtual environment and install the package with plotting extras
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install ".[viz]"

# ==========================================
# Stage 2: Runtime (headless execution)
# ==========================================
FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534 AS runtime

# Headless matplotlib backend and the venv on PATH
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PATH="/opt/venv/bin:$PATH"

# Runtime system libraries for matplotlib and bidi text shaping
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libfontconfig1 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the virtual environment and project files from the builder stage
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

# Default command: analyse the bundled example end to end (smoke test)
CMD ["truss-analysis", "analyze", "examples/example1.json", "--check-buckling"]
