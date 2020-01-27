#!/usr/bin/env bash

cd "${BASH_SOURCE%/*}" || exit

FROM_NAME=${1:-"main"}

if [ "$FROM_NAME" == "main" ]; then
	FROM=./data/tu
	TARGETS="./ds_splits ./libs/gnn-comparison/DATA"
elif [ "$FROM_NAME" == "comp" ]; then
	FROM=./libs/gnn-comparison/DATA
	TARGETS="./ds_splits ./data/tu"
elif [ "$FROM_NAME" == "splits" ]; then
	FROM=./ds_splits
	TARGETS="./data/tu ./libs/gnn-comparison/DATA"
else
	echo "Unknown dataset location name '$FROM_NAME'."
	echo "Use either 'main' (default), 'comp' or 'splits'."
	exit 1
fi

SPLIT_NAMES=$(ls $FROM/*/processed/*_splits.json)
echo "Found $(wc -l <<< "$SPLIT_NAMES") splits in $FROM."
echo "Copying to $TARGETS."
for from_split in $SPLIT_NAMES; do
	for target in $TARGETS; do
		to_split=$(sed "s#$FROM#$target#g" <<< "$from_split")
		mkdir -p $(dirname $to_split)
		cp $from_split $to_split
		echo "Synced: $from_split -> $to_split"
	done
done

echo "Done."
