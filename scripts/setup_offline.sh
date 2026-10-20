#!/usr/bin/env bash
# One-time setup for fully offline research.
#   ./scripts/setup_offline.sh            # installs omniresearch + pulls a default model
#   MODEL=qwen2.5:14b ./scripts/setup_offline.sh
set -euo pipefail
MODEL="${MODEL:-llama3.1:8b}"
cd "$(dirname "$0")/.."

echo ">> installing omniresearch (with PDF support)"
python3 -m pip install -e ".[pdf]"

if ! command -v ollama >/dev/null 2>&1; then
  cat <<MSG
>> Ollama is not installed. Install it from https://ollama.com/download (Linux: curl -fsSL https://ollama.com/install.sh | sh)
   then re-run this script, or point omniresearch at any OpenAI-compatible local server:
     omniresearch ask --backend openai --model <id> "..."
MSG
  exit 0
fi

if ! curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo ">> starting ollama serve in the background"
  (ollama serve >/dev/null 2>&1 &) ; sleep 3
fi

echo ">> pulling model $MODEL (do this while you still have internet)"
ollama pull "$MODEL"

echo ">> optional: index your own documents so research works with no network"
echo "   omniresearch index ~/papers ~/books ~/notes"
echo ">> check everything:"
omniresearch doctor || true
echo ">> try it:"
echo "   omniresearch ask --offline \"How does soil organic carbon affect drought resilience?\""
