#!/bin/bash
# validate.sh
# Goes in ROOT folder: /Users/simransota/meta/rl_agent/validate.sh
# CREATE this as a new file.
#
# Run this before submitting to check everything is ready.
# Usage:
#   chmod +x validate.sh
#   ./validate.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS="${GREEN}✓ PASS${NC}"
FAIL="${RED}✗ FAIL${NC}"
WARN="${YELLOW}⚠ WARN${NC}"

echo "======================================"
echo "  rxgym Pre-Submission Validator"
echo "======================================"
echo ""

ERRORS=0

# ── Check 1: Required files exist ──────────────────────────────────────────
echo "Checking required files..."

required_files=(
    "openenv.yaml"
    "models.py"
    "inference.py"
    "server/app.py"
    "server/rl_agent_environment.py"
    "server/agents.py"
    "server/Dockerfile"
    "server/requirements.txt"
)

for f in "${required_files[@]}"; do
    if [ -f "$f" ]; then
        echo -e "  $PASS  $f exists"
    else
        echo -e "  $FAIL  $f MISSING"
        ERRORS=$((ERRORS + 1))
    fi
done

echo ""

# ── Check 2: openenv.yaml has tasks ────────────────────────────────────────
echo "Checking openenv.yaml..."

if grep -q "phase_i_dosing" openenv.yaml; then
    echo -e "  $PASS  Task 1 (phase_i_dosing) found"
else
    echo -e "  $FAIL  Task 1 (phase_i_dosing) missing from openenv.yaml"
    ERRORS=$((ERRORS + 1))
fi

if grep -q "allometric_scaling" openenv.yaml; then
    echo -e "  $PASS  Task 2 (allometric_scaling) found"
else
    echo -e "  $FAIL  Task 2 (allometric_scaling) missing from openenv.yaml"
    ERRORS=$((ERRORS + 1))
fi

if grep -q "combo_ddi" openenv.yaml; then
    echo -e "  $PASS  Task 3 (combo_ddi) found"
else
    echo -e "  $FAIL  Task 3 (combo_ddi) missing from openenv.yaml"
    ERRORS=$((ERRORS + 1))
fi

echo ""

# ── Check 3: Python imports work ───────────────────────────────────────────
echo "Checking Python imports..."

python_check() {
    local module=$1
    local label=$2
    if uv run python -c "import $module" 2>/dev/null; then
        echo -e "  $PASS  $label"
    else
        echo -e "  $FAIL  $label (run: uv sync)"
        ERRORS=$((ERRORS + 1))
    fi
}

python_check "pydantic"   "pydantic"
python_check "openai"     "openai"
python_check "numpy"      "numpy"
python_check "fastapi"    "fastapi"
python_check "openenv"    "openenv-core"

echo ""

# ── Check 4: Server starts ─────────────────────────────────────────────────
echo "Checking server starts..."

# Start server in background
uv run uvicorn server.app:app --host 0.0.0.0 --port 8765 &
SERVER_PID=$!
sleep 3   # wait for server to boot

# Test reset endpoint
if curl -s -X POST http://localhost:8765/reset | grep -q "dose_level"; then
    echo -e "  $PASS  /reset returns valid observation"
else
    echo -e "  $FAIL  /reset did not return expected fields"
    ERRORS=$((ERRORS + 1))
fi

# Test step endpoint
STEP_RESPONSE=$(curl -s -X POST http://localhost:8765/step \
    -H "Content-Type: application/json" \
    -d '{"next_dose": 2.0, "cohort_size": 3, "escalate": true}')

if echo "$STEP_RESPONSE" | grep -q "reward"; then
    echo -e "  $PASS  /step returns reward"
else
    echo -e "  $FAIL  /step did not return reward"
    ERRORS=$((ERRORS + 1))
fi

if echo "$STEP_RESPONSE" | grep -q "dlt_count"; then
    echo -e "  $PASS  /step returns dlt_count"
else
    echo -e "  $FAIL  /step did not return dlt_count"
    ERRORS=$((ERRORS + 1))
fi

# Test state endpoint
if curl -s http://localhost:8765/state | grep -q "step_count"; then
    echo -e "  $PASS  /state returns step_count"
else
    echo -e "  $FAIL  /state did not return expected fields"
    ERRORS=$((ERRORS + 1))
fi

# Kill test server
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo ""

# ── Check 5: Docker builds ─────────────────────────────────────────────────
echo "Checking Docker build..."

if command -v docker &>/dev/null; then
    if docker build -t rxgym-test:latest -f server/Dockerfile . --quiet 2>/dev/null; then
        echo -e "  $PASS  Docker build succeeded"
        docker rmi rxgym-test:latest --force &>/dev/null
    else
        echo -e "  $FAIL  Docker build failed (run: docker build -f server/Dockerfile .)"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "  $WARN  Docker not found — install Docker Desktop"
fi

echo ""

# ── Check 6: inference.py log format ──────────────────────────────────────
echo "Checking inference.py format..."

if grep -q "\[START\]" inference.py; then
    echo -e "  $PASS  [START] log format found"
else
    echo -e "  $FAIL  [START] missing from inference.py"
    ERRORS=$((ERRORS + 1))
fi

if grep -q "\[STEP\]" inference.py; then
    echo -e "  $PASS  [STEP] log format found"
else
    echo -e "  $FAIL  [STEP] missing from inference.py"
    ERRORS=$((ERRORS + 1))
fi

if grep -q "\[END\]" inference.py; then
    echo -e "  $PASS  [END] log format found"
else
    echo -e "  $FAIL  [END] missing from inference.py"
    ERRORS=$((ERRORS + 1))
fi

if grep -q "API_BASE_URL" inference.py; then
    echo -e "  $PASS  API_BASE_URL env variable used"
else
    echo -e "  $FAIL  API_BASE_URL missing from inference.py"
    ERRORS=$((ERRORS + 1))
fi

if grep -q "MODEL_NAME" inference.py; then
    echo -e "  $PASS  MODEL_NAME env variable used"
else
    echo -e "  $FAIL  MODEL_NAME missing from inference.py"
    ERRORS=$((ERRORS + 1))
fi

echo ""

# ── Final result ───────────────────────────────────────────────────────────
echo "======================================"
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}ALL CHECKS PASSED — Ready to submit!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Push to GitHub:  git add . && git commit -m 'rxgym submission' && git push"
    echo "  2. Deploy to HF:    huggingface-cli upload your-username/rxgym . --repo-type=space"
    echo "  3. Submit URL:      https://huggingface.co/spaces/your-username/rxgym"
else
    echo -e "${RED}$ERRORS CHECK(S) FAILED — Fix before submitting${NC}"
fi
echo "======================================"
