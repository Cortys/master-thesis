#!/usr/bin/env bash

USER="-u $(id -u):$(id -g)"
CUDA_ENV=""

if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
	CUDA_ENV="-e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

if [ -z "$LTAG_CONTAINER_NAME" ]; then
	LTAG_CONTAINER_NAME="ltag"
fi

docker exec -it $USER $CUDA_ENV $(docker ps -aqf "name=^$LTAG_CONTAINER_NAME\$") python3 ./ltag/evaluate_datasets.py $@ \
	| grep --line-buffered -vE \
	"BaseCollectiveExecutor::StartAbort|IteratorGetNext|Shape/|Shape_[0-9]+/"
