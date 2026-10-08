#!/bin/bash
# pg-mcp local database bootstrap: start cluster, import fixtures, verify
set -e

pg_ctlcluster 14 main start 2>/dev/null || true
sleep 2

su postgres -c "psql -qc \"ALTER USER postgres PASSWORD 'postgres';\""

cd /mnt/e/big-model-project/geektime-bootcamp-ai/w5/pg-mcp/fixtures
for f in 01_small_db.sql 02_medium_db.sql 03_large_db.sql; do
  echo "=== importing $f ==="
  su postgres -c "psql -q -v ON_ERROR_STOP=1 -f '$f'" 2>&1 | grep -E "ERROR|FATAL" || echo "OK: $f"
done

echo "=== databases ==="
su postgres -c "psql -t -c 'select datname from pg_database where not datistemplate order by 1'"
