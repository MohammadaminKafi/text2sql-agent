FROM mcr.microsoft.com/mssql/server:2022-latest

# ─── download AdventureWorks backup ──────────────────────────────────────────
ARG BAK_URL=https://github.com/Microsoft/sql-server-samples/releases/download/adventureworks/AdventureWorks2022.bak
USER root
RUN apt-get update && apt-get install -y curl gnupg2 ca-certificates && \
    mkdir -p /var/opt/mssql/backup && \
    curl -L "$BAK_URL" -o /var/opt/mssql/backup/AdventureWorks2022.bak && \
    \
    # ----- add Microsoft repo (Debian 12 style) & install tools -------------
    curl -sSL https://packages.microsoft.com/keys/microsoft.asc \
        | gpg --dearmor -o /etc/apt/trusted.gpg.d/microsoft.gpg && \
    echo "deb [arch=amd64 signed-by=/etc/apt/trusted.gpg.d/microsoft.gpg] \
        https://packages.microsoft.com/debian/12/prod bookworm main" \
        > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y \
        mssql-tools unixodbc-dev && \
    ln -sf /opt/mssql-tools/bin/sqlcmd /usr/local/bin/sqlcmd && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ─── restore helper ──────────────────────────────────────────────────────────
COPY init_db.sh /usr/local/bin/init_db.sh
RUN chmod +x /usr/local/bin/init_db.sh

USER mssql
ENTRYPOINT ["/usr/local/bin/init_db.sh"]