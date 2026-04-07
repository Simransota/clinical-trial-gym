#!/usr/bin/env bash
set -euo pipefail

PING_URL="${1:-}"
REPO_DIR="${2:-$(pwd)}"

if [[ -z "${PING_URL}" ]]; then
  echo "usage: $0 <ping_url> [repo_dir]" >&2
  exit 2
fi

echo "[1/5] checking required files"
test -f "${REPO_DIR}/openenv.yaml"
test -f "${REPO_DIR}/inference.py"
test -f "${REPO_DIR}/Dockerfile"
test -f "${REPO_DIR}/README.md"

echo "[2/5] validating HF Space reachability"
curl -fsSL "${PING_URL}/health" >/dev/null

echo "[3/5] building Docker image"
docker build -t rxgym-submission "${REPO_DIR}" >/dev/null

echo "[4/5] validating OpenEnv manifest"
python - <<PY
from pathlib import Path
import yaml

manifest = yaml.safe_load(Path("${REPO_DIR}/openenv.yaml").read_text())
assert manifest["name"] == "rxgym"
assert len(manifest["tasks"]) >= 3
for task in manifest["tasks"]:
    assert task["difficulty"] in {"easy", "medium", "hard"}
print("openenv.yaml ok")
PY

echo "[5/5] baseline smoke test"
python - <<PY
from pathlib import Path
src = Path("${REPO_DIR}/inference.py").read_text()
assert "[START]" in Path("${REPO_DIR}/rl_agent/inference.py").read_text()
assert "OpenAI" in Path("${REPO_DIR}/rl_agent/inference.py").read_text()
print("baseline script structure ok")
PY

echo "validation checks passed"
