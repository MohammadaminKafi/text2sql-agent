# ---------- Build stage ------------------------------------------------------
FROM python:3.12-slim AS base

# Install system deps for the SQL Server ODBC driver
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        gcc \
        libffi-dev \
        libpq-dev \
        libssl-dev \
        python3-dev \
        curl \
        git \
        unzip \
        make \
        gnupg2 \
        unixodbc-dev && \
    # Add Microsoft GPG key the correct way for Debian 12+
    curl -sSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /etc/apt/trusted.gpg.d/microsoft.gpg && \
    echo "deb [arch=amd64 signed-by=/etc/apt/trusted.gpg.d/microsoft.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql18 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
#RUN pip install --index-url https://mirrors.aliyun.com/pypi/simple/ --use-deprecated=legacy-resolver -r requirements.txt
RUN pip install -r requirements.txt

# Copy source
COPY . /app

# Expose Flask port
EXPOSE 44488

# Default command
CMD ["python", "main.py"]