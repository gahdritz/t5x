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

import flan.v2.mixtures

# A directory of TFrecords for each mixture
CACHE_DIR = os.environ["FLAN_CACHE_DIR"]
if(len(CACHE_DIR) == 0):
    raise ValueError("FLAN_CACHE_DIR must be specified")

FEATURE_MAP = {
    "_task_name": tf.io.FixedLenFeature([], tf.string),
    "_task_source": tf.io.FixedLenFeature([], tf.string),
    "_template_type": tf.io.FixedLenFeature([], tf.string),
    "inputs_pretokenized": tf.io.FixedLenFeature([], tf.string),
    "targets_pretokenized": tf.io.FixedLenFeature([], tf.string),
    "_template_idx": tf.io.FixedLenFeature([], tf.int64),
    "inputs": tf.io.FixedLenFeature([], tf.int64),
    "targets": tf.io.FixedLenFeature([], tf.int64),
}

T5_VOCAB_SIZE = 32128
CACHE_OUTPUT_FEATURES = {
    'inputs':
        seqio.Feature(
            vocabulary=seqio.PassThroughVocabulary(T5_VOCAB_SIZE),
        ),
    'targets':
        seqio.Feature(
            vocabulary=seqio.PassThroughVocabulary(T5_VOCAB_SIZE)
        )
}

##############################################################
##### Instantiate the submixtures with each template style
##############################################################

# ZSOPT, FSOPT, ZSNOOPT, FSNOOPT are template styles.
# ZS means a zero-shot prompt, FS means a few-shot prompt
# OPT means the answer options for tasks with multiple choice answers are included in the template
# NOOPT means the answer options for tasks with multiple choice answers are NOT included in the template

cache_tfrecords = [f for f in os.listdir(CACHE_DIR) if f.endswith(".tfrecord")]
unique_cache_tfrecords = set([f.rsplit('_', 1)[0] for f in cache_tfrecords])

in_cache = lambda name: any([name in f for f in unique_cache_tfrecords])

if(in_cache("cot_submix")):
    seqio.TaskRegistry.add(
        'cot_submix',
        seqio.TFExampleDataSource({"train": f"{CACHE_DIR}/cot_submix*"}, FEATURE_MAP),
        output_features=CACHE_OUTPUT_FEATURES,
        preprocessors=None,
        postprocessors=None,
        metric_fns=None,
    )
else:
    seqio.MixtureRegistry.add(
        'cot_submix',
        tasks=[
            ('cot_zsopt', 1),    # mixing weight = 50%
            ('cot_fsopt', 1),    # mixing weight = 50%
        ])

if(in_cache("dialog_submix")):
    seqio.TaskRegistry.add(
        'dialog_submix',
        seqio.TFExampleDataSource({"train": f"{CACHE_DIR}/dialog_submix*"}, FEATURE_MAP),
        output_features=CACHE_OUTPUT_FEATURES,
        preprocessors=None,
        postprocessors=None,
        metric_fns=None,
    )
else:
    seqio.MixtureRegistry.add(
        'dialog_submix',
        tasks=[
            ('dialog_zsopt', 1),    # mixing weight = 50%
            ('dialog_fsopt', 1),    # mixing weight = 50%
        ])


if(in_cache("niv2_zsopt")):
    print("IN HERE!!!!! IN HERE!!!!!!!!")
    seqio.TaskRegistry.add(
        'niv2_zsopt',
        seqio.TFExampleDataSource({"train": f"{CACHE_DIR}/niv2_submix*"}, FEATURE_MAP),
        output_features=CACHE_OUTPUT_FEATURES,
    )
else:
    pass
#    seqio.MixtureRegistry.add(
#        'niv2_submix',
#        tasks=[
#            ('niv2_zsopt', 1),    # mixing weight = 50%
#            ('niv2_fsopt', 1),    # mixing weight = 50%
#        ])


if(in_cache("flan2021_submix")):
    seqio.TaskRegistry.add(
        'flan2021_submix',
        seqio.TFExampleDataSource({"train": f"{CACHE_DIR}/flan2021_submix*"}, FEATURE_MAP),
        output_features=CACHE_OUTPUT_FEATURES,
        preprocessors=None,
        postprocessors=None,
        metric_fns=None,
    )
else:
    seqio.MixtureRegistry.add(
        'flan2021_submix',
        tasks=[
            ('flan_zsopt', 1),      # mixing weight = 25%
            ('flan_fsopt', 1),      # mixing weight = 25%
            ('flan_zsnoopt', 1),    # mixing weight = 25%
            ('flan_fsnoopt', 1),    # mixing weight = 25%
        ])


if(in_cache("t0_submix")):
    seqio.TaskRegistry.add(
        't0_submix',
        seqio.TFExampleDataSource({"train": f"{CACHE_DIR}/t0_submix*"}, FEATURE_MAP),
        output_features=CACHE_OUTPUT_FEATURES,
        preprocessors=None,
        postprocessors=None,
        metric_fns=None,
    )
else:
    seqio.MixtureRegistry.add(
        't0_submix',
        tasks=[
            ('t0_zsopt', 1),      # mixing weight = 25%
            ('t0_fsopt', 1),      # mixing weight = 25%
            ('t0_zsnoopt', 1),    # mixing weight = 25%
            ('t0_fsnoopt', 1),    # mixing weight = 25%
        ])


# Define the Final Flan Collection Mixture
seqio.MixtureRegistry.add(
    'flan2022_submix',
    tasks=[
        ('flan2021_submix', 0.4),  # mixing weight = 40%
        ('t0_submix', 0.32),       # mixing weight = 32%
        ('niv2_zsopt', 0.2),      # mixing weight = 20%
        ('cot_submix', 0.05),      # mixing weight = 5%
        ('dialog_submix', 0.03),   # mixing weight = 3%
    ])
