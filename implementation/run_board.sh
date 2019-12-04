#!/usr/bin/env bash

USER="-u $(id -u):$(id -g)"

if [ "$1" == "root" ]; then
	USER="-u 0:0"
fi

echo "Starting board at http://localhost:6006/"
docker exec -it $USER $(docker ps -aqf "name=ltag") tensorboard --logdir /logs --bind_all
