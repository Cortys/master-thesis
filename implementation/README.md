# LTAG Implementation

This is the reference implementation of the WL2 inspired LTAG graph classifier.

## Build

The implementation can be run in a Tensorflow container.
Other architectures are implemented in PyTorch and must be run in a separate container.
Since CUDA is required in both containers, the Nvidia container runtime should be installed.

To build the containers execute:
```bash
./run.sh rebuild tensorflow
./run.sh rebuild torch
```
This might take a while.

To sync the train/test/validation splits between the Tensorflow and PyTorch implementations, the [`./sync_datasets.sh`](./sync_datasets.sh) script should then be executed.
It copies over the splits from [`ds_repo`](./ds_repo) to the dataset repos of the Tensorflow and PyTorch implementations.

## How to use

Containers can be started via `./run.sh [tensorflow (default) | torch]`.
A bash shell can be attached to a running container via `./attach_shell.sh`.

If the Tensorflow container is running (for the main implementation): 
- A TensorBoard instance can be launched via `./run_board.sh`.
- The datasets can be downloaded and preprocessed via `./prepare_datasets.sh`.
- Evaluations can be started via `./evaluate_datasets.sh`.

If the PyTorch container is running (for the comparative implementations):
- The datasets can be downloaded and preprocessed via `./prepare_comp_datasets.sh`.
- Evaluations can be started via `./evaluate_comp_datasets.sh`.
