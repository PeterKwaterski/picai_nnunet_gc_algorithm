#!/usr/bin/env bash
set -euo pipefail

docker build \
  --file Dockerfile.v2 \
  --tag silverdisk/picai_nnunet_v2_processor:latest \
  .
