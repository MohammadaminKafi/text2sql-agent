# Backend Dockerfile
# Builds and runs the FastAPI backend, installing deps from both core and backend.

FROM python:3.13-slim

# Env
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (pyodbc needs unixODBC; basic build tools for any wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg ca-certificates \
        build-essential \
        gcc \
        g++ \
        unixodbc \
        unixodbc-dev \
    && mkdir -p /usr/share/keyrings /etc/apt/sources.list.d \
    && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies from both core and backend
COPY src/core/requirements.txt /tmp/requirements/core.txt
COPY src/backend/requirements.txt /tmp/requirements/backend.txt
RUN python -m pip install --upgrade pip \
 && pip install -r /tmp/requirements/core.txt \
 && pip install -r /tmp/requirements/backend.txt

# Copy the source code (backend depends on core)
COPY src /app/src

# Make the project importable
ENV PYTHONPATH=/app/src

EXPOSE 8000

# Start the FastAPI app with auto-reload for development
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app/src/backend", "--reload-dir", "/app/src/core"]
