#!/bin/bash
set -e

/opt/mssql/bin/sqlservr &     # start SQL Server in the background
pid=$!

# ---- wait until server answers --------------------------------------
until /opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P "$SA_PASSWORD" \
        -Q "SELECT 1" &>/dev/null; do
    echo "💤  waiting for SQL Server..."
    sleep 3
done

echo "🗄️  restoring AdventureWorks2022 if necessary …"
DB_EXISTS=$(/opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P "$SA_PASSWORD" \
            -h-1 -Q "SET NOCOUNT ON; SELECT COUNT(*) FROM sys.databases WHERE name='AdventureWorks2022';")
if [[ "$DB_EXISTS" == "0" ]]; then
  /opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P "$SA_PASSWORD" -Q "
    RESTORE DATABASE AdventureWorks2022
    FROM DISK = '/var/opt/mssql/backup/AdventureWorks2022.bak'
    WITH MOVE 'AdventureWorks2022'     TO '/var/opt/mssql/data/AdventureWorks2022.mdf',
         MOVE 'AdventureWorks2022_Log' TO '/var/opt/mssql/data/AdventureWorks2022_log.ldf',
         REPLACE, STATS = 5;"
  echo "✅  AdventureWorks2022 restored!"
else
  echo "ℹ️  AdventureWorks2022 already present – skipping restore."
fi

wait $pid                      # keep SQL Server in foreground