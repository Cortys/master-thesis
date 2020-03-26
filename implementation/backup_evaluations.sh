#!/usr/bin/env bash

cd "${BASH_SOURCE%/*}" || exit

TO=${1:-"$HOME/ltag_backup"}

TO_TF="$TO/tf/"
TO_TORCH="$TO/torch/"

echo "Backing up TF evaluation results to $TO_TF..."
rsync -zarvm --info=progress2 --include "*/" --include "*.json" --include "*.txt" --exclude "*" "./evaluations/" "$TO_TF"

echo "Backing up Torch evaluation results to $TO_TORCH..."
rsync -zarvm --info=progress2 --include "*/" --include "*.json" --include "*.log" --exclude "*" "./libs/gnn-comparison/RESULTS/" "$TO_TORCH"

echo "Done."
