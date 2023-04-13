#!/bin/bash

if [[ $# != 0 ]]; then
    echo "usage: create_tfrecords.sh"
    exit
fi

source venv/bin/activate
source env_setup.sh

TASK_ID=$SLURM_ARRAY_TASK_ID
TASK_COUNT=$SLURM_ARRAY_TASK_COUNT

python3 scripts/cache_scripts/cache_to_tfrecords.py \
    new_flan_cache \
    flan_cache_tfrecords \
    --examples_per_shard 100000 \
    --worker_id $SLURM_ARRAY_TASK_ID \
    --worker_count $SLURM_ARRAY_TASK_COUNT
