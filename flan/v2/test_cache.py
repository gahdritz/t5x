import copy
import functools
import json
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import seqio
import tensorflow as tf

import flan.v2.final_mixtures_cached

selected_mixture = seqio.get_mixture_or_task('niv2_zsopt')

# 2. Example use cases to use just all chain-of-thought templates together:
# selected_mixture = seqio.get_mixture_or_task('cot_submix')

# 3. Example use cases to use the full Flan Collection:
# selected_mixture = seqio.get_mixture_or_task('flan2022_submix')

# If you're using Seqio, we suggest caching your mixture as they take a while to generate.
# If you want to read out the post-processed examples into a file, we suggest using the
# sample_fn below to collect 1 epoch of data, according to our mixing rates.
INPUT_SEQ_LEN = 2056
TARGET_SEQ_LEN = 512
dataset = selected_mixture.get_dataset(
    sequence_length={"inputs": INPUT_SEQ_LEN, "targets": TARGET_SEQ_LEN},
    num_epochs=1,
    shuffle=True,
    copy_pretokenized=True,
    # The passthrough features let you track the source/task/template metadata for the example
    passthrough_features=["_template_idx", "_task_source", "_task_name", "_template", "_template_type"]
)

# To read out the data you can do something like this:
save_data = []
source_counter = defaultdict(lambda: 0)
NUM_SAMPLES = 100
# If you would like to take min(1 epoch, NUM_SAMPLES) then use dataset.take(NUM_SAMPLES)
# Or if you would like to gather a full epoch, simply `enumerate(dataset)` until completion.
for i, ex in enumerate(dataset.take(NUM_SAMPLES)):
    source_counter[ex["_task_source"].numpy()] += 1
    save_data.append((ex["inputs_pretokenized"].numpy().decode(),
                      ex["inputs"].numpy(),
                      ex["targets_pretokenized"].numpy().decode()))

print(f"Data Submixture Counts: {source_counter}")

print(save_data)
for e in save_data:
    print(len(e[0]))
    print(len(e[1]))
