#!/usr/bin/env sh
set -eu

MODE="${1:-api}"

require_env() {
  VAR_NAME="$1"
  eval "VAL=\${$VAR_NAME:-}"
  if [ -z "$VAL" ]; then
    echo "Missing required environment variable: $VAR_NAME" >&2
    exit 2
  fi
}

case "$MODE" in
  api)
    shift || true
    exec uvicorn rl_agent.server.app:app --host 0.0.0.0 --port "${PORT:-8000}" "$@"
    ;;
  inference)
    shift || true
    exec python /app/inference.py "$@"
    ;;
  report)
    shift || true
    require_env REPORT_DRUG_NAME
    require_env REPORT_DRUG_SMILES
    require_env REPORT_ANIMAL_DOSE_MGKG
    set -- python /app/analysis/generate_drug_report.py \
      --name "${REPORT_DRUG_NAME}" \
      --smiles "${REPORT_DRUG_SMILES}" \
      --animal-dose "${REPORT_ANIMAL_DOSE_MGKG}" \
      --source-species "${REPORT_SOURCE_SPECIES:-rat}" \
      "$@"
    if [ -n "${REPORT_OUTDIR:-}" ]; then
      set -- "$@" --outdir "${REPORT_OUTDIR}"
    fi
    exec "$@"
    ;;
  shell)
    shift || true
    exec sh "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
