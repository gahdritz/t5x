import argparse
import os

import numpy as np
import tensorflow as tf

INT_LEN = 8
TEXT_FIELDS = set([
    "_task_name",
    "_task_source",
    "_template_type",
    "inputs_pretokenized",
    "targets_pretokenized",
])
INT_FIELDS = set([
    "_template_idx",
    "inputs",
    "targets",
])

def parse_examples(database, bounds_file):
    db = open(database, 'rb')
    bounds_file = open(bounds_file, 'rb')

    while(True):
        bounds = bounds_file.read(2 * INT_LEN)
        if(len(bounds) == 0):
            break

        start, l = bounds[:INT_LEN], bounds[INT_LEN:]
        start = int.from_bytes(start, 'little')
        l = int.from_bytes(l, 'little')

        db.seek(start)
        example_bytes = db.read(l)

        example = {}
        while(len(example_bytes) > 0):
            k_len_bytes, v_len_bytes = example_bytes[:INT_LEN], example_bytes[INT_LEN:2*INT_LEN]
            example_bytes = example_bytes[2*INT_LEN:]
            k_len, v_len = int.from_bytes(k_len_bytes, 'little'), int.from_bytes(v_len_bytes, 'little')
            k, v = example_bytes[:k_len], example_bytes[k_len:k_len+v_len]
            example_bytes = example_bytes[k_len+v_len:]
            k = k.decode('utf-8')
            if(k in TEXT_FIELDS):
                v = v.decode('utf-8')
            elif(k in INT_FIELDS):
                v = np.frombuffer(v, dtype=np.int32)
            else:
                raise Exception(f"Unknown field type: {k}")

            example[k] = v

        yield example


def main(args):
    for f in os.listdir(args.cache_dir):
        if(f.endswith('_bounds.db')):
            continue

        database = os.path.join(args.cache_dir, f)
        bounds_file = os.path.join(args.cache_dir, f[:-3] + '_bounds.db')

        example_gen = parse_examples(database, bounds_file)

        for i, ex in enumerate(example_gen):
            if(i % 1000 == 0):
                print(f"{i}...")

            if(i % args.examples_per_shard == 0):
                shard = i // args.examples_per_shard
                tfrecords_filename = os.path.join(args.tfrecords_dir, f"{f[:-3]}_{shard}.tfrecord")
                writer = tf.io.TFRecordWriter(tfrecords_filename)

            def to_feature(k, v):
                if(k in TEXT_FIELDS):
                    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v.encode('utf-8')]))
                elif(k in INT_FIELDS):
                    return tf.train.Feature(int64_list=tf.train.Int64List(value=v.tolist()))
                else:
                    raise Exception(f"Unknown field type: {k}")

            example = tf.train.Example(features=tf.train.Features(feature={
                k: to_feature(k,v)
                for k,v in ex.items()
            }))

            writer.write(example.SerializeToString())




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cache_dir', type=str)
    parser.add_argument('tfrecords_dir', type=str)
    parser.add_argument('--examples_per_shard', type=int, default=10000)
    args = parser.parse_args()

    main(args)
