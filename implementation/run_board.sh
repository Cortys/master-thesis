#!/usr/bin/env bash

USER="-u $(id -u):$(id -g)"

LOGDIR="/${1:-logs}"

if [ -z "$LTAG_CONTAINER_NAME" ]; then
	LTAG_CONTAINER_NAME="ltag"
fi

echo "Starting board at http://localhost:6006/"
docker exec -it $USER $(docker ps -aqf "name=^$LTAG_CONTAINER_NAME\$") tensorboard --logdir $LOGDIR --bind_all
