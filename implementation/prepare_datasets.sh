#!/usr/bin/env bash

USER="-u $(id -u):$(id -g)"

if [ -z "$LTAG_CONTAINER_NAME" ]; then
	LTAG_CONTAINER_NAME="ltag"
fi

docker exec -it $USER $(docker ps -aqf "name=^$LTAG_CONTAINER_NAME\$") python3 ./ltag/prepare_datasets.py $@
