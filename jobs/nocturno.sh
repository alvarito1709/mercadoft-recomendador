#!/usr/bin/env bash
# Job nocturno del recomendador. Lo dispara el timer de systemd (jobs/systemd/) o cron.
# Variables: RECO_ORIGEN=gs://<bucket>/<carpeta>  RECO_BASE=/var/lib/mft
set -euo pipefail
cd "$(dirname "$0")/.."
exec nice -n 10 uv run --no-dev python -m mft.job_nocturno "$@"
