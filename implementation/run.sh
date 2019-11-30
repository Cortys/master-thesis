#!/usr/bin/env bash

cd "${BASH_SOURCE%/*}" || exit

JUPYTER_TOKEN=${JUPYTER_TOKEN:-$(cat JUPYTER_TOKEN)}
JUPYTER_URL="http://localhost:8888/?token=$JUPYTER_TOKEN"

if [ ! -z "$(docker ps -aqf "name=ltag")" ]; then
	echo "Notebook already started. See ${JUPYTER_URL}" >&2
	exit 1
fi

VARIANT=${1:-tensorflow}
ARGS=""

if [ "$VARIANT" == "tensorflow" ]; then
	ARGS="-u $(id -u):$(id -g)"
fi

trap 'kill %1; exit 0' SIGINT
trap 'kill -TERM %1; exit 0' SIGTERM

docker build . -t ltag/ltag-$VARIANT -f Dockerfile.$VARIANT
echo "Starting Notebook at ${JUPYTER_URL} ..."
echo "Using additional ARGS='$ARGS' with variant $VARIANT."
echo "Type \"rm\" to clean logs."
docker run --runtime=nvidia --rm --name ltag \
	-p 8888:8888 \
	-p 6006:6006 \
	-v $(pwd)/src:/ltag \
	-v $(pwd)/logs:/logs \
	-v $(pwd)/libs:/libs \
	-e "JUPYTER_TOKEN=$JUPYTER_TOKEN" \
	$ARGS ltag/ltag-$VARIANT &

while true; do
	read in

	if [ "$in" == "rm" ]; then
		rm -r  ./logs/[^.]* 2> /dev/null && echo "Removed logs." || echo "No logs to remove."
	fi
done
