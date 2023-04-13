import argparse
import copy
import functools
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import seqio
import tensorflow as tf

import flan.v2.mixtures
from flan.v2 import constants

def main(args):
    ##############################################################
    ##### Instantiate the submixtures with each template style
    ##############################################################
    
    # ZSOPT, FSOPT, ZSNOOPT, FSNOOPT are template styles.
    # ZS means a zero-shot prompt, FS means a few-shot prompt
    # OPT means the answer options for tasks with multiple choice answers are included in the template
    # NOOPT means the answer options for tasks with multiple choice answers are NOT included in the template
    
    seqio.MixtureRegistry.add(
        'cot_submix',
        tasks=[
            ('cot_zsopt', 1),    # mixing weight = 50%
            ('cot_fsopt', 1),    # mixing weight = 50%
        ])
    
    seqio.MixtureRegistry.add(
        'dialog_submix',
        tasks=[
            ('dialog_zsopt', 1),    # mixing weight = 50%
            ('dialog_fsopt', 1),    # mixing weight = 50%
        ])
    
    seqio.MixtureRegistry.add(
        'niv2_submix',
        tasks=[
            ('niv2_zsopt', 1),    # mixing weight = 50%
            ('niv2_fsopt', 1),    # mixing weight = 50%
        ])
    
#    seqio.MixtureRegistry.add(
#        'flan2021_submix',
#        tasks=[
#            ('flan_zsopt', 1),      # mixing weight = 25%
#            ('flan_fsopt', 1),      # mixing weight = 25%
#            ('flan_zsnoopt', 1),    # mixing weight = 25%
#            ('flan_fsnoopt', 1),    # mixing weight = 25%
#        ])
    
    seqio.MixtureRegistry.add(
        't0_submix',
        tasks=[
            *[(f't0_zsopt_{i}', 1) for i in range(constants.SUBMIX_SPLITS["T0"])],
            *[(f't0_fsopt_{i}', 1) for i in range(constants.SUBMIX_SPLITS["T0"])],
            *[(f't0_zsnoopt_{i}', 1) for i in range(constants.SUBMIX_SPLITS["T0"])],
            *[(f't0_fsnoopt_{i}', 1) for i in range(constants.SUBMIX_SPLITS["T0"])],
        ])
    
    # Define the Final Flan Collection Mixture
#    seqio.MixtureRegistry.add(
#        'flan2022_submix',
#        tasks=[
#            ('flan2021_submix', 0.4),  # mixing weight = 40%
#            ('t0_submix', 0.32),       # mixing weight = 32%
#            ('niv2_submix', 0.2),      # mixing weight = 20%
#            ('cot_submix', 0.05),      # mixing weight = 5%
#            ('dialog_submix', 0.03),   # mixing weight = 3%
#        ])
    
    
    ##############################################################
    ##### See 3 Examples of Mixtures or Submixtures you can try
    ##############################################################

    passthrough_features =  [
        "_template_idx", 
    ]
    if(args.mixture_name != "t0_submix"):
        passthrough_features.extend([
            "_template", 
            "_template_type"
        ])
    elif(args.mixture_name not in ["flan2021_submix", "t0_submix"]):
        passthrough_features.extend([
            "_task_source", 
            "_task_name",
        ])

    if(args.mixture_name == "t0_submix"):
        splits = [
            *[f't0_fsopt_{i}' for i in range(constants.SUBMIX_SPLITS["T0"])],
        ]
        
        import random
        args.mixture_name = random.choice(splits)

    if(args.mixture_name == "flan2021_submix"):
        #splits = [
        #    *[f'flan_fsopt_{i}' for i in range(constants.SUBMIX_SPLITS["FLAN"])],
        #    *[f'flan_fsnoopt_{i}' for i in range(constants.SUBMIX_SPLITS["FLAN"])],
        #]
        splits = ['flan_fsopt_2']
        
        import random
        args.mixture_name = random.choice(splits)

    print(f"Selecting mixture {args.mixture_name}...")
    selected_mixture = seqio.get_mixture_or_task(args.mixture_name)
    
    seqio.add_global_cache_dirs(["/n/holyscratch01/jfrankle_lab/Lab/gahdritz/flan_cache"])

    # If you're using Seqio, we suggest caching your mixture as they take a while to generate.
    # If you want to read out the post-processed examples into a file, we suggest using the
    # sample_fn below to collect 1 epoch of data, according to our mixing rates.
    INPUT_SEQ_LEN = args.input_seq_len
    TARGET_SEQ_LEN = args.target_seq_len
    dataset = selected_mixture.get_dataset(
        sequence_length={"inputs": INPUT_SEQ_LEN, "targets": TARGET_SEQ_LEN},
        num_epochs=1,
        shuffle=True,
        copy_pretokenized=True,
        # The passthrough features let you track the source/task/template metadata for the example
        passthrough_features=passthrough_features,
        use_cached=(args.mixture_name == "t0_submix")
    )
    
    save_data = []
    NUM_SAMPLES = args.max_examples
    for i, ex in enumerate(dataset.take(NUM_SAMPLES)):
        if(i % 1000 == 0):
            print(f"{i}...")
        save_data.append(ex)
   
    name_base = f"{args.cache_dir}/{args.mixture_name}_{INPUT_SEQ_LEN}_{TARGET_SEQ_LEN}_{''.join(str(time.time()).split('.'))}"
    db_filename = f"{name_base}.db"
    bounds_filename = f"{name_base}_bounds.db"
    db_file = open(db_filename, 'wb')
    bounds_file = open(bounds_filename, 'wb')
    bytes_written = 0
    INT_LEN = 8
    for ex in save_data:
        local_bytes_written = 0
        for k,v in ex.items():
            v = v.numpy()
            k_bytes = k.encode('utf-8')
            if(type(v) is bytes):
                v_bytes = v
            else:
                v_bytes = v.tobytes()

            k_len_bytes = len(k_bytes).to_bytes(INT_LEN, 'little')
            v_len_bytes = len(v_bytes).to_bytes(INT_LEN, 'little')

            to_write = [
                k_len_bytes,
                v_len_bytes,
                k_bytes,
                v_bytes,
            ]

            for b in to_write:
                db_file.write(b)

            local_bytes_written += sum([len(b) for b in to_write])

        bytes_written_bytes = bytes_written.to_bytes(INT_LEN, 'little')
        local_bytes_written_bytes = local_bytes_written.to_bytes(INT_LEN, 'little')
        bounds_file.write(bytes_written_bytes)
        bounds_file.write(local_bytes_written_bytes)

        bytes_written += local_bytes_written
   
    db_file.close()
    bounds_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mixture_name", type=str)
    parser.add_argument("cache_dir", type=str)
    parser.add_argument("--input_seq_len", type=int, default=2048)
    parser.add_argument("--target_seq_len", type=int, default=512)
    parser.add_argument("--max_examples", type=int, default=1000000)

    args = parser.parse_args()

    main(args)
