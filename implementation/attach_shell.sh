#!/usr/bin/env bash

USER="-u $(id -u):$(id -g)"

if [ $1 == "root" ]; then
	USER="-u 0:0"
fi

docker exec -it $USER $(docker ps -aqf "name=ltag") bash
