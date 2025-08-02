FROM mcr.microsoft.com/mssql/server:2022-latest

# --- grab AdventureWorks backup (≈ 260 MB) --------------------------
ARG BAK_URL=https://github.com/Microsoft/sql-server-samples/releases/download/adventureworks/AdventureWorks2022.bak
USER root
RUN apt-get update && apt-get install -y curl \
 && mkdir -p /var/opt/mssql/backup \
 && curl -L $BAK_URL -o /var/opt/mssql/backup/AdventureWorks2022.bak \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- restore helper ---------------------------------------------------
COPY init_db.sh /usr/local/bin/init_db.sh
RUN chmod +x /usr/local/bin/init_db.sh

USER mssql
ENTRYPOINT ["/usr/local/bin/init-db.sh"]