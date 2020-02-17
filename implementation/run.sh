#!/usr/bin/env bash

cd "${BASH_SOURCE%/*}" || exit

if [ -z "$LTAG_CONTAINER_NAME" ]; then
	LTAG_CONTAINER_NAME="ltag"
fi

if [ -z "$JUPYTER_PORT" ]; then
	JUPYTER_PORT=8888
fi

JUPYTER_TOKEN=${JUPYTER_TOKEN:-$(cat JUPYTER_TOKEN)}
JUPYTER_URL="http://localhost:$JUPYTER_PORT/?token=$JUPYTER_TOKEN"

if [ ! -z "$(docker ps -aqf "name=^$LTAG_CONTAINER_NAME\$")" ]; then
	echo "Notebook already started in container ${LTAG_CONTAINER_NAME}. See ${JUPYTER_URL}" >&2
	exit 1
fi

DEFAULT_VARIANT=${LTAG_VARIANT:-"tensorflow"}
VARIANT=${1:-$DEFAULT_VARIANT}
ARGS=""
REBUILD=""

if [ "$1" == "rebuild" ]; then
	REBUILD=1
	VARIANT=${2:-$DEFAULT_VARIANT}
fi

if [ "$2" == "rebuild" ]; then
	REBUILD=1
fi

if [ "$VARIANT" == "tensorflow" ]; then
	ARGS="-p 6006:6006 -u $(id -u):$(id -g) -e TF_FORCE_GPU_ALLOW_GROWTH=$TF_FORCE_GPU_ALLOW_GROWTH"
fi

trap 'kill %1; exit 0' SIGINT
trap 'kill -TERM %1; exit 0' SIGTERM

echo "Using container variant: $VARIANT."

if [ "$REBUILD" == "1" ]; then
	echo "Building container..."
	docker build . -t ltag/ltag-$VARIANT -f Dockerfile.$VARIANT
fi

echo "Starting Notebook at ${JUPYTER_URL} ..."
echo "Using additional ARGS='$ARGS' with variant $VARIANT and container name $LTAG_CONTAINER_NAME."
echo "Type \"rm\" to clean logs."

mkdir -p logs
mkdir -p evaluations
mkdir -p libs
mkdir -p data

docker run --gpus all --rm --name $LTAG_CONTAINER_NAME \
	-p $JUPYTER_PORT:8888 \
	-v $(pwd)/src:/ltag \
	-v $(pwd)/logs:/logs \
	-v $(pwd)/evaluations:/evaluations \
	-v $(pwd)/libs:/libs \
	-v $(pwd)/data:/data \
	-e "JUPYTER_TOKEN=$JUPYTER_TOKEN" \
	$ARGS ltag/ltag-$VARIANT &

while true; do
	read in

	if [ "$in" == "rm" ]; then
		rm -r ./logs/[^.]* 2> /dev/null && echo "Removed logs." || echo "No logs to remove."
	fi
done
