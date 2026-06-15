#!/usr/bin/env bash
set -euo pipefail

PROFILE_NAME="${PROFILE_NAME:-gpt55}"
MODEL_NAME="${MODEL_NAME:-gpt-5.5}"
PROVIDER_NAME="${PROVIDER_NAME:-custom}"
BASE_URL="${BASE_URL:-https://a-ocnfniawgw.cn-shanghai.fcapp.run}"
CODEX_HOME_DIR="${CODEX_HOME:-$HOME/.codex}"

mkdir -p "$CODEX_HOME_DIR"

CONFIG_FILE="$CODEX_HOME_DIR/${PROFILE_NAME}.config.toml"

cat > "$CONFIG_FILE" <<EOF
model_provider = "$PROVIDER_NAME"
model = "$MODEL_NAME"
model_reasoning_effort = "high"
disable_response_storage = true

[model_providers]
[model_providers.$PROVIDER_NAME]
name = "$PROVIDER_NAME"
wire_api = "responses"
requires_openai_auth = true
base_url = "$BASE_URL"
EOF

echo "Wrote Codex profile: $CONFIG_FILE"
echo
echo "Start Codex with:"
echo "  codex --profile $PROFILE_NAME"
echo
echo "Or test non-interactively:"
echo "  codex --profile $PROFILE_NAME exec \"hello\""
