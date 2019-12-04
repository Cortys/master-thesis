#!/usr/bin/env bash

source /etc/bash.bashrc

trap 'kill %1' SIGINT
trap 'kill -TERM %1' SIGTERM

jupyter lab --notebook-dir=/ltag --ip 0.0.0.0 --no-browser --allow-root &
wait $!
