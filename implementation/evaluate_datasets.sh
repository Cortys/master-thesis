#!/usr/bin/env bash

USER="-u $(id -u):$(id -g)"

docker exec -it $USER $(docker ps -aqf "name=ltag") \
	python3 ./ltag/evaluate_datasets.py \
	| grep --line-buffered -vE \
	"BaseCollectiveExecutor::StartAbort|IteratorGetNext|Shape/"
