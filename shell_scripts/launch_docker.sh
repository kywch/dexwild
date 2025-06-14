#!/bin/bash

# Ensure the script is run with sudo
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)." 1>&2
   exit 1
fi

CONTAINER_NAME="manus_container"

# Run the Docker container
echo "Starting the Docker container..."
sudo docker run --name $CONTAINER_NAME -p 8000:8000 --privileged -v /dev:/dev -v /run/udev:/run/udev -i boardd/manussdk:v0 /bin/bash
if [ $? -ne 0 ]; then
    echo "Failed to start the Docker container."
    exit 1
fi

echo "Docker Launched"