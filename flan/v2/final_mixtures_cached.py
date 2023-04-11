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

from flan.v2 import (
    constants,
    mixtures,
)


# A directory of TFrecords for each mixture
CACHE_DIR = os.environ["FLAN_CACHE_DIR"]
if(len(CACHE_DIR) == 0):
    raise ValueError("FLAN_CACHE_DIR must be specified")

USE_CACHE = True

FEATURE_MAP = {
#    "_task_name": tf.io.FixedLenFeature([], tf.string),
#    "_task_source": tf.io.FixedLenFeature([], tf.string),
#    "_template_type": tf.io.FixedLenFeature([], tf.string),
#    "inputs_pretokenized": tf.io.FixedLenFeature([], tf.string),
#    "targets_pretokenized": tf.io.FixedLenFeature([], tf.string),
#    "_template_idx": tf.io.FixedLenFeature([], tf.int64),
    "inputs": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    "targets": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
}

T5_VOCAB_SIZE = 32128 
CACHE_OUTPUT_FEATURES = {
    'inputs':
        seqio.Feature(
            vocabulary=seqio.PassThroughVocabulary(T5_VOCAB_SIZE),
            add_eos=False,
            dtype=tf.int32,
        ),
    'targets':
        seqio.Feature(
            vocabulary=seqio.PassThroughVocabulary(T5_VOCAB_SIZE),
            add_eos=False,
            dtype=tf.int32,
        )
}


def cast_to_int32(dataset):
    def _cast(ex):
        d = {
            "inputs": tf.cast(ex["inputs"], tf.int32),
            "targets": tf.cast(ex["targets"], tf.int32),
        }
        ex.update(d)
        return ex
    return dataset.map(_cast)


def purge_from_seqio(name):
    seqio.TaskRegistry.remove(name)
    seqio.MixtureRegistry.remove(name)


def create_cached_task(name):
    purge_from_seqio(name)

    seqio.TaskRegistry.add(
        name,
        seqio.TFExampleDataSource({"train": f"{CACHE_DIR}/{name}*"}, FEATURE_MAP),
        output_features=CACHE_OUTPUT_FEATURES,
        preprocessors=[cast_to_int32],
        postprocess_fn=None,
        metric_fns=None,
    )


cache_tfrecords = [f for f in os.listdir(CACHE_DIR) if f.endswith(".tfrecord")]
unique_cache_tfrecords = set([f.rsplit('_', 1)[0] for f in cache_tfrecords])

in_cache = lambda name: any([name in f for f in unique_cache_tfrecords])

##############################################################
##### Instantiate the submixtures with each template style
##############################################################

# ZSOPT, FSOPT, ZSNOOPT, FSNOOPT are template styles.
# ZS means a zero-shot prompt, FS means a few-shot prompt
# OPT means the answer options for tasks with multiple choice answers are included in the template
# NOOPT means the answer options for tasks with multiple choice answers are NOT included in the template

if(USE_CACHE and in_cache("cot_submix")):
    create_cached_task("cot_submix")
else:
    seqio.MixtureRegistry.add(
        'cot_submix',
        tasks=[
            ('cot_zsopt', 1),    # mixing weight = 50%
            ('cot_fsopt', 1),    # mixing weight = 50%
        ])

if(USE_CACHE and in_cache("dialog_submix")):
    create_cached_task("dialog_submix")
else:
    seqio.MixtureRegistry.add(
        'dialog_submix',
        tasks=[
            ('dialog_zsopt', 1),    # mixing weight = 50%
            ('dialog_fsopt', 1),    # mixing weight = 50%
        ])


if(USE_CACHE and in_cache("niv2_zsopt")):
    create_cached_task("niv2_zsopt")
else:
    pass
#    seqio.MixtureRegistry.add(
#        'niv2_submix',
#        tasks=[
#            ('niv2_zsopt', 1),    # mixing weight = 50%
#            ('niv2_fsopt', 1),    # mixing weight = 50%
#        ])


if(USE_CACHE and in_cache("flan2021_submix")):
    create_cached_task("flan2021_submix")
else:
    if(USE_CACHE and in_cache("flan_zsopt")):
        for name in ["flan_zsopt", "flan_fsopt", "flan_zsnoopt", "flan_fsnoopt"]:
            assert(in_cache(name))
            create_cached_task(name)

    seqio.MixtureRegistry.add(
        'flan2021_submix',
        tasks=[
            ('flan_zsopt', 1),      # mixing weight = 25%
            ('flan_fsopt', 1),      # mixing weight = 25%
            ('flan_zsnoopt', 1),    # mixing weight = 25%
            ('flan_fsnoopt', 1),    # mixing weight = 25%
        ])


if(USE_CACHE and in_cache("t0_submix")):
    create_cached_task("t0_submix")
else:
    if(USE_CACHE and in_cache("t0_zsopt")):
        for name in ["t0_zsopt", "t0_fsopt", "t0_zsnoopt", "t0_fsnoopt"]:
            assert(in_cache(name))
            create_cached_task(name)

    seqio.MixtureRegistry.add(
        't0_submix',
        tasks=[
            ('t0_zsopt', 1),      # mixing weight = 25%
            ('t0_fsopt', 1),      # mixing weight = 25%
            ('t0_zsnoopt', 1),    # mixing weight = 25%
            ('t0_fsnoopt', 1),    # mixing weight = 25%
        ])


# Define the final Flan Collection Mixture
seqio.MixtureRegistry.add(
    'flan2022_submix',
    tasks=[
        ('flan2021_submix', 0.4),  # mixing weight = 40%
        ('t0_submix', 0.32),       # mixing weight = 32%
        ('niv2_zsopt', 0.2),      # mixing weight = 20%
        ('cot_submix', 0.05),      # mixing weight = 5%
        ('dialog_submix', 0.03),   # mixing weight = 3%
    ])

#def bad(x):
#    for i in range(10):
#        if(f"_template_{i}_" in x):
#            return True
#
#    return False
#
#x = seqio.get_mixture_or_task("flan_fsopt")
#print(type(x))
#print(dir(x))
#
#print([x for x in seqio.TaskRegistry.names() if "bool_q" in x and "zero_shot" in x])
#dodo = [x for x in seqio.TaskRegistry.names() if "bool_q" in x and not bad(x)]
#print(dodo)
#seqio.MixtureRegistry.add(
#    "topo",
#    tasks=[(x, 1.0) for x in dodo],
#)
#
#print(dodo)
#
#selected_mixture = seqio.get_mixture_or_task("flan2022_submix")
###selected_mixture = seqio.get_mixture_or_task("glue_mnli_v2")
###selected_mixture = seqio.get_mixture_or_task("topo")
##
#INPUT_SEQ_LEN = 2056
#TARGET_SEQ_LEN = 512
#dataset = selected_mixture.get_dataset(
#    sequence_length={"inputs": INPUT_SEQ_LEN, "targets": TARGET_SEQ_LEN},
#    split='train',
#    #split='validation',
#    num_epochs=None,
#    shuffle=True,
#    copy_pretokenized=True,
#    trim_output_features=True,
#    shard_info=seqio.ShardInfo(index=0, num_shards=4),
#    seed=42,
#)
#
## To read out the data you can do something like this:
#save_data = []
#source_counter = defaultdict(lambda: 0)
#NUM_SAMPLES = 100
#c = 0
## If you would like to take min(1 epoch, NUM_SAMPLES) then use dataset.take(NUM_SAMPLES)
## Or if you would like to gather a full epoch, simply `enumerate(dataset)` until completion.
##for i, ex in enumerate(dataset.take(NUM_SAMPLES)):
#for i, ex in enumerate(dataset):
##    c += 1
##    if(c % 10000 == 0):
##        print(c)
#    save_data.append(ex)
#    print(f"inputs: {ex['inputs_pretokenized'].numpy().decode('utf-8')}")
#    print(f"targets: {ex['targets_pretokenized'].numpy().decode('utf-8')}")
#    print(ex['inputs'])
#    print(ex['targets'])
#    
##print(c)
