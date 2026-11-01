#!/bin/sh
# Run both benchmarks inside the backend container, then copy the reports out.
#
# From the project root, with the stack running (docker compose up -d):
#     sh backend/eval/run_all.sh                     # default Ollama models below
#     sh backend/eval/run_all.sh qwen2.5:7b qwen3:8b # your own list
# Reports land in backend/eval/results/ (Markdown + raw JSON): commit them.
set -e

MODELS="${*:-qwen2.5:7b qwen3:8b llama3.1:8b}"
OLLAMA_URL="${OLLAMA_URL:-http://host.docker.internal:11434/v1}"

model_args=""
for m in $MODELS; do model_args="$model_args --model $m"; done

echo "== Routing: mock + $MODELS (2 repeats each)"
docker compose exec -T backend python -m eval.routing --mock $model_args \
    --base-url "$OLLAMA_URL" --api-key ollama --repeats 2

echo "== Retrieval: 3 embedding models x 2 chunk sizes, through the API"
docker compose exec -T backend python -m eval.retrieval

docker compose cp backend:/app/eval/results/. backend/eval/results/
echo "Done: see backend/eval/results/"
