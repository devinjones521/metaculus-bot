#!/usr/bin/env bash
# Deploy metaculus-bot to the Hetzner box. Idempotent: rerun it after any change.
#
#   bash deploy/push.sh          code + units, then restart the pollers
#   bash deploy/push.sh --env    ...and copy .env first (first deploy, or a new key)
#
# Touches ONLY /opt/metaculus-bot, /var/log/metaculus-bot and its own
# metaculus-poll@ units. This box also serves live websites; nothing here goes
# near nginx, PM2 or cron.
set -euo pipefail

HOST="root@204.168.148.150"
KEY="$HOME/.ssh/id_ed25519"
ROOT="/opt/metaculus-bot"
SSH=(ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=15 "$HOST")

cd "$(dirname "$0")/.."

# Never deploy red.
uv run python -m bot.verify > /dev/null || { echo "verify is red — not deploying."; exit 1; }

# Code only, never data/: the box's logs are its own evidence, and a laptop copy
# would overwrite them.
tar -czf - --exclude=__pycache__ bot deploy pyproject.toml uv.lock README.md LICENSE \
  | "${SSH[@]}" "mkdir -p $ROOT/data/metaculus /var/log/metaculus-bot && tar -xzf - -C $ROOT"

if [[ "${1:-}" == "--env" ]]; then
  # Over stdin, so the keys never appear on a command line or in `ps`.
  "${SSH[@]}" "umask 077 && cat > $ROOT/.env" < .env
  echo ".env copied (mode 600)."
fi

mapfile -t UNITS < <(grep -vE '^[[:space:]]*(#|$)' deploy/units)

"${SSH[@]}" bash -s -- "$ROOT" "${UNITS[@]}" <<'REMOTE'
set -euo pipefail
ROOT=$1; shift
UV=/root/.local/bin/uv
test -x "$UV" || curl -LsSf https://astral.sh/uv/install.sh | sh
test -s "$ROOT/.env" || { echo "no $ROOT/.env on the box — rerun with --env"; exit 1; }
cd "$ROOT" && "$UV" sync --frozen --no-dev -q
install -m 644 deploy/metaculus-poll@.service /etc/systemd/system/
systemctl daemon-reload
for unit in "$@"; do
  systemctl enable -q "$unit"
  systemctl restart "$unit"
done
sleep 8
for unit in "$@"; do
  printf '%-40s %s, restarts=%s\n' "$unit" \
    "$(systemctl is-active "$unit")" "$(systemctl show "$unit" -p NRestarts --value)"
done
REMOTE

echo "deployed. Liveness: uv run python -m bot.health"
