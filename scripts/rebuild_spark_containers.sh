#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="docker-compose-infra.yml"

SERVICES=(
  spark-master
  spark-worker-1
  spark-worker-2
  spark-worker-3
  spark-history
)

# Detect docker access: try without sudo first, then with sudo
if docker info >/dev/null 2>&1; then
  DOCKER_CMD="docker compose"
elif sudo docker info >/dev/null 2>&1; then
  echo "Note: docker requires sudo on this host; falling back to sudo for docker commands."
  DOCKER_CMD="sudo docker compose"
else
  echo "ERROR: docker daemon not accessible (neither plain docker nor sudo docker worked)."
  echo "Either install/start Docker or add your user to the docker group:"
  echo "  sudo usermod -aG docker \$USER  &&  newgrp docker  # then re-open shell"
  exit 1
fi

echo "=> Building images for: ${SERVICES[*]}"
$DOCKER_CMD -f "$COMPOSE_FILE" build "${SERVICES[@]}"

echo "=> Recreating and starting containers for: ${SERVICES[*]}"
$DOCKER_CMD -f "$COMPOSE_FILE" up -d --force-recreate --no-deps "${SERVICES[@]}"

echo "=> Done."
