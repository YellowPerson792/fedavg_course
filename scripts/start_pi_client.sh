#!/usr/bin/env bash
set -euo pipefail

CLIENT_ID="${1:-pi0}"
CLIENT_INDEX="${2:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src"
python -m fedavg.client --config "$ROOT/configs/pi_client.yaml" --client-id "$CLIENT_ID" --client-index "$CLIENT_INDEX"
