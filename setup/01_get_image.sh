#!/usr/bin/env bash
# Obtain the SGLang 26.06 container image.
#   docker : pulls the image tag.
#   enroot : imports a squashfs once (avoids per-run registry-import races).
set -euo pipefail
cd "$(dirname "$0")/.."
source ./config.env

case "$CONTAINER_ENGINE" in
  docker)
    echo "[01] docker pull $IMAGE_TAG"
    docker pull "$IMAGE_TAG"
    ;;
  enroot)
    if [ -f "$IMAGE_SQSH" ]; then
      echo "[01] squashfs already present: $IMAGE_SQSH"
    else
      echo "[01] enroot import -> $IMAGE_SQSH"
      enroot import -o "$IMAGE_SQSH" "docker://nvcr.io#nvidia/sglang:26.06-py3"
    fi
    ;;
  *)
    echo "unknown CONTAINER_ENGINE=$CONTAINER_ENGINE (use docker or enroot)" >&2
    exit 2
    ;;
esac
echo "[01] done."
