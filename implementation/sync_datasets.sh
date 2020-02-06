#!/usr/bin/env bash

cd "${BASH_SOURCE%/*}" || exit

FROM_NAME=${1:-"splits"}

function sync_datasets() {
	local FROM=$1
	local TARGETS=$2
	local PATTERN=$3
	local MATCH_NAMES=$(ls -d $FROM/$PATTERN)

	local from_match
	local to_match
	local target

	echo "Found $(wc -l <<< "$MATCH_NAMES") matches for $FROM/$PATTERN."
	echo "Copying to $TARGETS."
	for from_match in $MATCH_NAMES; do
		for target in $TARGETS; do
			to_match=$(sed "s|$FROM|$target|g" <<< "$from_match")
			[ -e $to_match ] && rm -r $to_match
			mkdir -p $(dirname $to_match)
			cp -RT --remove-destination $from_match $to_match
			echo "Synced: $from_match -> $to_match"
		done
	done
}

if [ "$FROM_NAME" == "main" ]; then
	TU_FROM="./data/tu"
	SYN_FROM="./data/synthetic"
	TU_TARGETS="./ds_splits/tu ./libs/gnn-comparison/DATA"
	SYN_TARGETS="./ds_splits/synthetic ./libs/gnn-comparison/SYN_DATA"
elif [ "$FROM_NAME" == "comp" ]; then
	TU_FROM="./libs/gnn-comparison/DATA"
	SYN_FROM="./libs/gnn-comparison/SYN_DATA"
	TU_TARGETS="./ds_splits/tu ./data/tu"
	SYN_TARGETS="./ds_splits/synthetic ./data/synthetic"
elif [ "$FROM_NAME" == "splits" ]; then
	TU_FROM="./ds_splits/tu"
	SYN_FROM="./ds_splits/synthetic"
	TU_TARGETS="./data/tu ./libs/gnn-comparison/DATA"
	SYN_TARGETS="./data/synthetic ./libs/gnn-comparison/SYN_DATA"
else
	echo "Unknown dataset location name '$FROM_NAME'."
	echo "Use either 'main', 'comp' or 'splits' (default)."
	exit 1
fi

echo "Syncing tu dataset splits..."
# sync_datasets "$TU_FROM" "$TU_TARGETS" "*/processed/*_splits.json"
echo "Synced tu dataset splits."
echo

echo "Syncing synthetic dataset splits..."
# sync_datasets "$SYN_FROM" "$SYN_TARGETS" "*/processed/*_splits.json"
echo "Synced synthetic dataset splits."
echo
echo "Syncing synthetic dataset raw encodings..."
# sync_datasets "$SYN_FROM" "$SYN_TARGETS" "*/raw"
echo "Synced synthetic dataset raw encodings."
echo

echo "Done."
