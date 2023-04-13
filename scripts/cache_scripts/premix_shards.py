import argparse
import os

import seqio
import tensorflow as tf

from flan.v2 import (
    constants,
    mixtures,
)

"""
There's a limit to how many tasks you can put in a single seqio mixture, so
it's necessary to premix the submixtures we sharded to prevent too many 
tasks from being created at once during the caching step.
"""


EXCLUDE = set([
    "flan_fsopt_2",
])

UNSPLIT = set([
    "flan_zsopt",
    "flan_zsnoopt",
])

EXCEPTIONS = {
    "t0_zsopt": 4,
    "t0_zsnoopt": 4,
    "t0_fsnoopt": 4,
}


def get_submix_coefficients():     
    all_mixtures = seqio.MixtureRegistry.names()
    submix_mixture_coefficients = {}
    for mix, no_submix_splits in constants.SUBMIX_SPLITS.items():
        unsorted_mixtures = [m for m in all_mixtures if m.startswith(f"{mix.lower()}_")]
        if(no_submix_splits is not None): 
            unique = set([m.rsplit('_', 1)[0] for m in unsorted_mixtures])
            for u in unique:
                no_submix_splits_copy = no_submix_splits
                if(u in EXCEPTIONS):
                    no_submix_splits = EXCEPTIONS[u]
                    
    
                tups = []
                for i in range(no_submix_splits):
                    submix_name = f"{u}_{i}"
                    submix_split = seqio.get_mixture_or_task(submix_name)
    
                    if(submix_name in EXCLUDE):
                        continue 
    
                    tups.append((submix_name, submix_split.total_rate))
    
                submix_mixture_coefficients[u] = tups
    
                no_submix_splits = no_submix_splits_copy
        else:
            for submix in unsorted_mixtures:
                submix_mixture_coefficients[submix] = [(submix, 1)]

    return submix_mixture_coefficients


SUBMIX_MIXTURE_COEFFICIENTS = get_submix_coefficients()

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


FEATURE_MAP = {
#    "_task_name": tf.io.FixedLenFeature([], tf.string),
#    "_task_source": tf.io.FixedLenFeature([], tf.string),
#    "_template_type": tf.io.FixedLenFeature([], tf.string),
    "inputs_pretokenized": tf.io.FixedLenFeature([], tf.string),
    "targets_pretokenized": tf.io.FixedLenFeature([], tf.string),
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


def create_cached_task(name, cache_dir):
    purge_from_seqio(name)

    def _cache_task(n):
        seqio.TaskRegistry.add(
            n,
            seqio.TFExampleDataSource({"train": f"{cache_dir.rstrip('/')}/{n}*"}, FEATURE_MAP),
            output_features=CACHE_OUTPUT_FEATURES,
            preprocessors=[cast_to_int32],
            postprocess_fn=None,
            metric_fns=None,
        )

    if(name in SUBMIX_MIXTURE_COEFFICIENTS and not name in UNSPLIT):
        for submix_name, _ in SUBMIX_MIXTURE_COEFFICIENTS[name]:
            purge_from_seqio(submix_name)
            _cache_task(submix_name)
    
        seqio.MixtureRegistry.add(
            name,
            tasks=SUBMIX_MIXTURE_COEFFICIENTS[name],
        )
    else:
        _cache_task(name)


def main(args):
    if(args.num_workers is not None):
        worker_id = args.worker_id
        num_workers = args.num_workers
    else:
        worker_id = 0
        num_workers = 1

    for name, shards in SUBMIX_MIXTURE_COEFFICIENTS.items():
        if(name in UNSPLIT or len(shards) == 1):
            continue

        create_cached_task(name, args.cache_dir)

        selected_mixture = seqio.get_mixture_or_task(name)

        INPUT_SEQ_LEN = 2056
        TARGET_SEQ_LEN = 512
        dataset = selected_mixture.get_dataset(
            sequence_length={"inputs": INPUT_SEQ_LEN, "targets": TARGET_SEQ_LEN},
            split='train',
            num_epochs=1,
            shuffle=True,
            trim_output_features=True,
            shard_info=seqio.ShardInfo(index=worker_id, num_shards=num_workers),
            seed=42,
        )

        for i, ex in enumerate(dataset):
            if(i % 1000 == 0):
                print(f"{i}...")

            if(i % args.examples_per_shard == 0):
                shard = i // args.examples_per_shard
                tfrecords_filename = os.path.join(args.output_dir, f"{name}_unified_{shard}_{worker_id}_{num_workers}.tfrecord")
                writer = tf.io.TFRecordWriter(tfrecords_filename)

            def to_feature(v):
                return tf.train.Feature(int64_list=tf.train.Int64List(value=v.numpy().tolist()))

            example = tf.train.Example(features=tf.train.Features(feature={
                k: to_feature(v)
                for k,v in ex.items()
            }))

            writer.write(example.SerializeToString())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--worker_id", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--examples_per_shard", type=int, default=10000)
    args = parser.parse_args()

    if((args.worker_id is None) ^ (args.num_workers is None)):
        raise ValueError("Must specify both worker_id and num_workers")

    main(args)

    
    
