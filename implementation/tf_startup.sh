#!/usr/bin/env bash

source /etc/bash.bashrc

trap 'kill %1 %2' SIGINT
trap 'kill -TERM %1 %2' SIGTERM

tensorboard --logdir /logs --bind_all &
TB_PID=$!

jupyter lab --notebook-dir=/ltag --ip 0.0.0.0 --no-browser --allow-root &
JUPYTER_PID=$!

wait $TB_PID $JUPYTER_PID
