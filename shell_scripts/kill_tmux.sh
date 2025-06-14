#!/bin/bash

tmux kill-session -t record_data

# ensure the Docker container is stopped & removed
CONTAINER_NAME="manus_container"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true