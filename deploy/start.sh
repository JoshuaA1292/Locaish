#!/bin/sh
set -e
# ClickHouse first: zero-config server, data lives and dies with the
# instance -- every cold start reloads the shot table from the baked dumps.
cd /var/lib/clickhouse
clickhouse server > /var/log/clickhouse.log 2>&1 &
for i in $(seq 1 60); do
  clickhouse client --query "SELECT 1" >/dev/null 2>&1 && break
  sleep 1
done
clickhouse client --query "SELECT 1" >/dev/null 2>&1 || {
  echo "clickhouse did not come up"; tail -20 /var/log/clickhouse.log; exit 1; }
python -c "from locaish import warehouse; warehouse.ensure_schema(); warehouse.ensure_plans_schema()"
clickhouse client --query "INSERT INTO locaish.shot_setups FORMAT Native" < /app/chdata/shot_setups.native
# The plans dump ships empty when the gallery starts with a clean slate;
# an empty INSERT is not worth risking the boot on.
[ -s /app/chdata/shot_plans.native ] \
  && clickhouse client --query "INSERT INTO locaish.shot_plans FORMAT Native" < /app/chdata/shot_plans.native
echo "shot table loaded: $(clickhouse client --query 'SELECT count() FROM locaish.shot_setups') setups"
cd /app
exec locaish studio --no-open --showcase --root /app/rooms --max-points 800000
