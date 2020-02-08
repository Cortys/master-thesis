#!/usr/bin/env bash

USER="-u $(id -u):$(id -g)"

if [ -z "$LTAG_CONTAINER_NAME" ]; then
	LTAG_CONTAINER_NAME="ltag"
fi

echo "Preparing dataset encodings..."
docker exec -it $USER $(docker ps -aqf "name=^$LTAG_CONTAINER_NAME\$") python3 ./ltag/prepare_datasets.py $@
echo "Prepared dataset encodings."

OPTIND=1
gram=1

while getopts ":gpd:" opt; do
    case "$opt" in
	g)  gram=0
    	;;
    esac
done

if [ $gram == "0" ]; then
	echo "Not preparing 2-WL gram matrices."
	exit 0
fi

DATASETS=$(cat <<- END
	LWL2 noisy_triangle_classification_dataset ./data/synthetic
	LWL2 MUTAG ./data/tu
	LWL2 NCI1 ./data/tu
	LWL2 PROTEINS_full ./data/tu
	LWL2 DD ./data/tu
	LWL2 REDDIT-BINARY ./data/tu

	GWL2 noisy_triangle_classification_dataset ./data/synthetic
	GWL2 MUTAG ./data/tu
	GWL2 NCI1 ./data/tu
	GWL2 PROTEINS_full ./data/tu
	GWL2 DD ./data/tu
	GWL2 REDDIT-BINARY ./data/tu
END
)
parallelism=4

echo "Preparing 2-WL gram matrices..."
echo "$DATASETS" | xargs -n 3 --max-procs=$parallelism ./libs/glocalwl/globalwl
echo "Prepared 2-WL gram matrices."
