#!/bin/bash

if [[ $# != 1 ]]; then
    echo "usage: create_caches.sh <max>"
    exit
fi

MAX=$1

source venv/bin/activate
source env_setup.sh

TASK_ID=$SLURM_ARRAY_TASK_ID

NAME_ARRAY=("flan_zsopt" "flan_fsopt" "flan_zsnoopt" "flan_fsnoopt" "t0_submix")
NAME_ARRAY_LEN="${#NAME_ARRAY[@]}"
IDX=$(($TASK_ID % $NAME_ARRAY_LEN))
NAME="${NAME_ARRAY[$IDX]}"

for i in $(seq 0 1 ${MAX}); do
    python3 scripts/cache_scripts/create_cache.py $NAME
done
