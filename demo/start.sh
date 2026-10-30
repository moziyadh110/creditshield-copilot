#!/bin/sh
# Boot the demo container. Everything is rebuilt on each start (Cloud Run keeps no disk):
# Postgres, migrations, prompt registration (v1 becomes @production) and the Reg B index.
set -e
export PGDATA=/home/user/pgdata
rm -rf "$PGDATA"
initdb -D "$PGDATA" -U user --auth=trust > /dev/null
pg_ctl -D "$PGDATA" -o "-k /tmp -c listen_addresses=localhost" -l /home/user/postgres.log -w start
createdb -h localhost -U user copilot
alembic upgrade head
python -m app.llmops.prompts
python -m app.retrieval.ingest
exec uvicorn app.api.main:app --host 0.0.0.0 --port 7860 --proxy-headers
