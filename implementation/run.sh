#!/usr/bin/env bash

cd "${BASH_SOURCE%/*}" || exit

JUPYTER_TOKEN=${JUPYTER_TOKEN:-$(cat JUPYTER_TOKEN)}

VARIANT=${1:-tensorflow}

ARGS=""

if [ "$VARIANT" == "tensorflow" ]; then
	ARGS="-u $(id -u):$(id -g)"
fi

docker build . -t ltag/ltag-$VARIANT -f Dockerfile.$VARIANT
echo "Starting Notebook at http://localhost:8888/?token=$JUPYTER_TOKEN ..."
echo "Using additional ARGS='$ARGS' with variant $VARIANT."
docker run --runtime=nvidia --rm -p 8888:8888 --name ltag \
	-v $(pwd)/src:/ltag \
	-v $(pwd)/libs:/libs \
	-e "JUPYTER_TOKEN=$JUPYTER_TOKEN" \
	$ARGS ltag/ltag-$VARIANT
