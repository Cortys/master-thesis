#!/usr/bin/env bash

USER="-u $(id -u):$(id -g)"

if [ "$1" == "root" ]; then
	USER="-u 0:0"
fi

if [ -z "$LTAG_CONTAINER_NAME" ]; then
	LTAG_CONTAINER_NAME="ltag"
fi

docker exec -it $USER $(docker ps -aqf "name=$LTAG_CONTAINER_NAME") bash
