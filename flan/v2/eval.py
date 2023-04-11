from collections import defaultdict
import seqio

from flan.v2 import constants
from flan.v2 import postprocessors
from flan.v2 import preprocessors
from flan.v2 import task_configs_v1
from flan.v2 import templates
from t5.data import glue_utils
from t5.evaluation import metrics as t5_metrics
import t5x.contrib.gpu.scripts_gpu.seqio_tasks 


TASK_IDS = ["arc", "unified_qa_science_inst", "bool_q", "rte", "anli"]


for difficulty in ["Easy", "Challenge"]:
    name = f"arc_{difficulty.lower()}"
    patterns = templates.PATTERNS["arc"]
    #patterns = patterns[0:1]
    #formatter = preprocessors.get_formatter(patterns[0][0], patterns[0][1])
    formatter = preprocessors.get_batch_formatter(patterns)

    prep = [
        task_configs_v1._process_arc,
        task_configs_v1._filter_arc,
        preprocessors.format_options,
    ]
    prep += formatter
    prep += preprocessors.FLAN_TOKENIZE

    seqio.TaskRegistry.add(
        f"{name}_val",
        source=seqio.TfdsDataSource(
            tfds_name=f'ai2_arc/ARC-{difficulty}:1.0.0',
            splits={
                'validation': 'validation',
            }),
        preprocessors=prep,    
        postprocess_fn=None,
        output_features=constants.DEFAULT_OUTPUT_FEATURES,
        metric_fns=[t5_metrics.accuracy],
    )

name = "unified_qa_science_inst"
patterns = templates.PATTERNS[name]
#patterns = patterns[0:1]
#formatter = preprocessors.get_formatter(patterns[0][0], patterns[0][1])
formatter = preprocessors.get_batch_formatter(patterns)

prep = [
    preprocessors.filter_unified_qa_science_inst,
    preprocessors.unified_qa_science_inst,
    preprocessors.format_options,
]
prep += formatter
prep += preprocessors.FLAN_TOKENIZE

seqio.TaskRegistry.add(
    f"{name}_val",
    source=seqio.TfdsDataSource(
        tfds_name="unified_qa/ai2_science_middle:1.0.0",
        splits={
            'validation': 'validation',
        }),
    preprocessors=prep,    
    postprocess_fn=postprocessors.take_first_line,
    output_features=constants.DEFAULT_OUTPUT_FEATURES,
    metric_fns=[t5_metrics.accuracy],
)

name = "bool_q"
patterns = templates.PATTERNS[name]
#patterns = patterns[0:1]
#formatter = preprocessors.get_formatter(patterns[0][0], patterns[0][1])
formatter = preprocessors.get_batch_formatter(patterns)

prep = [
    task_configs_v1._process_boolq,
    preprocessors.format_options,
]
prep += formatter
prep += preprocessors.FLAN_TOKENIZE

seqio.TaskRegistry.add(
    f"{name}_val",
    source=seqio.TfdsDataSource(
        tfds_name="bool_q:1.0.0",
        splits={
            'validation': 'validation',
        }),
    preprocessors=prep,    
    postprocess_fn=None,
    output_features=constants.DEFAULT_OUTPUT_FEATURES,
    metric_fns=glue_utils.get_super_glue_metric('boolq'),
)

#def test(targets, predictions):
#    with open("here.txt", "w") as fp:
#        fp.write(str(targets))
#        fp.write('\n')
#        fp.write(str(predictions))
#    return t5_metrics.accuracy(targets, predictions)

for config_name in ['r1', 'r2', 'r3']:
    name = f"anli_{config_name}"
    patterns = templates.PATTERNS["anli"]
    #patterns = patterns[0:1]
    #formatter = preprocessors.get_formatter(patterns[0][0], patterns[0][1])
    formatter = preprocessors.get_batch_formatter(patterns)
    
    prep = [
        task_configs_v1._process_anli,
        preprocessors.format_options,
    ]
    prep += formatter
    prep += preprocessors.FLAN_TOKENIZE

    seqio.TaskRegistry.add(
        f"{name}_val",
        source=seqio.TfdsDataSource(
            tfds_name=f'anli/{config_name}:0.1.0',
            splits={
                'validation': 'validation',
            }),
        preprocessors=prep,    
        postprocess_fn=None,
        output_features=constants.DEFAULT_OUTPUT_FEATURES,
        #metric_fns=[test],
        metric_fns=[t5_metrics.accuracy],
    )
    
name = "rte"
patterns = templates.PATTERNS[name]
#patterns = patterns[0:1]
#formatter = preprocessors.get_formatter(patterns[0][0], patterns[0][1])
formatter = preprocessors.get_batch_formatter(patterns)

prep = [
    task_configs_v1._process_rte,
    preprocessors.format_options,
]
prep += formatter
prep += preprocessors.FLAN_TOKENIZE

seqio.TaskRegistry.add(
    f"{name}_val",
    source=seqio.TfdsDataSource(
        tfds_name="super_glue/rte:1.0.2",
        splits={
            'validation': 'validation',
        }),
    preprocessors=prep,    
    postprocess_fn=None,
    output_features=constants.DEFAULT_OUTPUT_FEATURES,
    metric_fns=glue_utils.get_super_glue_metric('rte'),
)

seqio.MixtureRegistry.add(
    'flan2022_val_mixture',
    tasks=[
        ('arc_easy_val', 1),
        ('arc_challenge_val', 1),
        ('unified_qa_science_inst_val', 1),
        ('bool_q_val', 1),
        ('anli_r1_val', 1),
        ('anli_r2_val', 1),
        ('anli_r3_val', 1),
        ('rte_val', 1),
    ]
)

#selected_mixture = seqio.get_mixture_or_task('flan2022_val_mixture')
##selected_mixture = seqio.get_mixture_or_task("glue_mnli_v2")
#
#INPUT_SEQ_LEN = 2056
#TARGET_SEQ_LEN = 512
#dataset = selected_mixture.get_dataset(
#    sequence_length={"inputs": INPUT_SEQ_LEN, "targets": TARGET_SEQ_LEN},
#    split='validation',
#    num_epochs=1,
#    shuffle=True,
#    copy_pretokenized=True,
#)
#
## To read out the data you can do something like this:
#save_data = []
#source_counter = defaultdict(lambda: 0)
#NUM_SAMPLES = 100
## If you would like to take min(1 epoch, NUM_SAMPLES) then use dataset.take(NUM_SAMPLES)
## Or if you would like to gather a full epoch, simply `enumerate(dataset)` until completion.
#for i, ex in enumerate(dataset.take(NUM_SAMPLES)):
#    save_data.append(ex)
#    #print(f"inputs: {ex['inputs_pretokenized']}")
#    #print(f"targets: {ex['targets_pretokenized']}")
#    #print(ex['inputs'])
#    #print(ex['targets'])
#    print(ex)
#
#print(len(save_data))
