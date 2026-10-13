
#!/usr/bin/env bash
set -Eeuo pipefail

: "${POSTGRES_USER:?PostgreSQL administrator user is missing}"
: "${POSTGRES_DB:?PostgreSQL database is missing}"
: "${POSTGRES_APP_USER:?POSTGRES_APP_USER is missing}"
: "${POSTGRES_APP_PASSWORD:?POSTGRES_APP_PASSWORD is missing}"

if [[ "$POSTGRES_APP_USER" == "$POSTGRES_USER" ]]; then
    echo "The application user must differ from the administrator user." >&2
    exit 1
fi

psql \
    --username "$POSTGRES_USER" \
    --dbname "$POSTGRES_DB" \
    --set=ON_ERROR_STOP=1 \
    --set=app_user="$POSTGRES_APP_USER" \
    --set=app_password="$POSTGRES_APP_PASSWORD" <<'SQL'
SELECT format(
    'CREATE ROLE %I LOGIN PASSWORD %L',
    :'app_user',
    :'app_password'
)
WHERE NOT EXISTS (
    SELECT 1
    FROM pg_roles
    WHERE rolname = :'app_user'
)
\gexec

SELECT format(
    'GRANT CONNECT ON DATABASE %I TO %I',
    current_database(),
    :'app_user'
)
\gexec

SELECT format(
    'GRANT USAGE, CREATE ON SCHEMA public TO %I',
    :'app_user'
)
\gexec
SQL

echo "PostgreSQL application user initialized."
