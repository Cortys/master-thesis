#!/usr/bin/env bash

USER="-u $(id -u):$(id -g)"

if [ -z "$LTAG_CONTAINER_NAME" ]; then
	LTAG_CONTAINER_NAME="ltag"
fi

docker exec -it $USER $(docker ps -aqf "name=^$LTAG_CONTAINER_NAME\$") bash /libs/gnn-comparison/evaluate_datasets.sh $@
