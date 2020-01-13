#!/usr/bin/env bash

USER="-u $(id -u):$(id -g)"

LOGDIR="/${1:-logs}"

echo "Starting board at http://localhost:6006/"
docker exec -it $USER $(docker ps -aqf "name=ltag") tensorboard --logdir $LOGDIR --bind_all
