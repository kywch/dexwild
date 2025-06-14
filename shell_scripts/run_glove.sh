#!/bin/bash

# Ensure the script is run with sudo
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)." 1>&2
   exit 1
fi

CONTAINER_NAME="manus_container"

# Define cleanup commands
cleanup() {
echo "Stopping and removing the Docker container..."
sudo docker stop "$CONTAINER_NAME"
sudo docker rm "$CONTAINER_NAME"
}

# Set the trap to call cleanup on EXIT
trap cleanup EXIT

# Commands to run inside the Docker container
echo "Running commands inside the container..."
sudo docker exec -it "$CONTAINER_NAME" bash -c "
   cd /home/manus/ManusSDK/SDKClient_Linux
    ./SDKClient_Linux.out
" 

echo "Manus Started"



