import argparse
import os

import seqio
import tensorflow as tf

from flan.v2 import mmlu


TEXT_FIELDS = set([
    "question",
    "options_",
    "answer",
    "_task_name",
    "_task_source",
    "_template_type",
    "inputs_pretokenized",
    "targets_pretokenized",
])
INT_FIELDS = set([
    "inputs",
    "targets",
])

REMOVE = [
    "options",
]


def main(args):
    examples = []
    for t in mmlu.MMLU_TASKS:
        selected_task = seqio.get_mixture_or_task(t)
        
        dataset = selected_task.get_dataset(
            sequence_length={"inputs": None, "targets": None},
            split='validation',
            num_epochs=1,
            trim_output_features=True,
        )
        
        for i, ex in enumerate(dataset):
            examples.append(ex)
  
    tfrecords_filename = os.path.join(args.out_dir, "mmlu.tfrecord")
    writer = tf.io.TFRecordWriter(tfrecords_filename)
    for i, ex in enumerate(examples):
        for k in REMOVE:
            ex.pop(k)

        def to_feature(k, v):
            if(k in TEXT_FIELDS):
                return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v.numpy()]))
            elif(k in INT_FIELDS):
                return tf.train.Feature(int64_list=tf.train.Int64List(value=v.numpy().tolist()))
            else:
                raise Exception(f"Unknown field type: {k}")

        example = tf.train.Example(features=tf.train.Features(feature={
            k: to_feature(k,v)
            for k,v in ex.items()
        }))

        writer.write(example.SerializeToString())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=str)

    args = parser.parse_args()

    main(args)
