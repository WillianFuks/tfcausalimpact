#!/usr/bin/env bash

set -euo pipefail

PYTHON_VERSION="${1:-3.13}"

case "${PYTHON_VERSION}" in
    3.8) TOX_ENV="py38-linux" ;;
    3.9) TOX_ENV="py39-linux" ;;
    3.10) TOX_ENV="py310-linux" ;;
    3.11) TOX_ENV="py311-linux" ;;
    3.12) TOX_ENV="py312-linux" ;;
    3.13) TOX_ENV="py313-linux" ;;
    *)
        echo "Unsupported Python version: ${PYTHON_VERSION}" >&2
        echo "Choose one of: 3.8, 3.9, 3.10, 3.11, 3.12, 3.13" >&2
        exit 1
        ;;
esac

docker run --rm \
    --mount "type=bind,source=$(pwd),target=/workspace" \
    --workdir /workspace \
    "python:${PYTHON_VERSION}-slim" \
    sh -c "python -m pip install --no-cache-dir tox && tox --workdir /tmp/tox -e ${TOX_ENV}"
