#!/usr/bin/env bash
# Smoke test for the absa entry point. Usage: scripts/smoke_test.sh COMMAND [ARGS...]
# Example: scripts/smoke_test.sh python -m absa
set -euo pipefail

good='{"review": "The battery is weak.", "aspect": "battery", "sentiment": "negative"}'
bad='{"review": "The battery is weak."}'

[[ "$("$@" --version)" == "absa "* ]] || { echo "unexpected --version output" >&2; exit 1; }
[[ "$(printf '%s\n' "$good" | "$@" validate)" == "ok" ]] || { echo "valid record rejected" >&2; exit 1; }
if printf '%s\n' "$bad" | "$@" validate 2>/dev/null; then
  echo "invalid record accepted" >&2
  exit 1
fi
echo "smoke test passed"
