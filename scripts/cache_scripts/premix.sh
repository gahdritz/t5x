#!/bin/bash

if [[ $# != 0 ]]; then
    echo "usage: premix.sh"
    exit
fi

source venv/bin/activate
source env_setup.sh

TASK_ID=$SLURM_ARRAY_TASK_ID
TASK_COUNT=$SLURM_ARRAY_TASK_COUNT

python3 scripts/cache_scripts/premix_shards.py \
    flan_cache_tfrecords \
    premixed_tfrecords \
    --examples_per_shard 100000 \
    --worker_id $SLURM_ARRAY_TASK_ID \
    --num_workers $SLURM_ARRAY_TASK_COUNT
