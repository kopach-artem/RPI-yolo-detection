#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

IMAGE_NAME="${IMAGE_NAME:-yolo-pi4:torch231}"
CONTAINER_NAME="${CONTAINER_NAME:-yolo-droidcam}"
ENV_FILE="${ENV_FILE:-.env}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed."
  echo "Run first:"
  echo "  ./install_docker_debian.sh"
  exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
  if [ -f ".env.example" ]; then
    cp .env.example "$ENV_FILE"
    echo "Created $ENV_FILE from .env.example"
    echo "Edit it first, then run again:"
    echo "  nano $ENV_FILE"
    exit 1
  else
    echo "Missing $ENV_FILE and .env.example"
    exit 1
  fi
fi

mkdir -p data/shots data/json data/alerts

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  echo "Docker image $IMAGE_NAME not found. Building..."
  docker build -t "$IMAGE_NAME" .
else
  echo "Docker image $IMAGE_NAME already exists. Skipping build."
fi

echo "Starting container..."
docker run --rm -it --ipc=host \
  --name "$CONTAINER_NAME" \
  -p 8080:8080 \
  --env-file "$ENV_FILE" \
  -v "$PWD:/work" \
  -w /work \
  "$IMAGE_NAME" \
  bash -lc "python -m eval.run_images_eval --subset wide  && python -m eval.metrics --subset wide && python -m eval.plots --subset wide"

