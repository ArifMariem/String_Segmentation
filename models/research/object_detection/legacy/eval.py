# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Evaluation executable for detection models.

This executable is used to evaluate DetectionModels. There are two ways of
configuring the eval job.

1) A single pipeline_pb2.TrainEvalPipelineConfig file maybe specified instead.
In this mode, the --eval_training_data flag may be given to force the pipeline
to evaluate on training data instead.

Example usage:
    ./eval \
        --logtostderr \
        --checkpoint_dir=path/to/checkpoint_dir \
        --eval_dir=path/to/eval_dir \
        --pipeline_config_path=pipeline_config.pbtxt

2) Three configuration files may be provided: a model_pb2.DetectionModel
configuration file to define what type of DetectionModel is being evaluated, an
input_reader_pb2.InputReader file to specify what data the model is evaluating
and an eval_pb2.EvalConfig file to configure evaluation parameters.

Example usage:
    ./eval \
        --logtostderr \
        --checkpoint_dir=path/to/checkpoint_dir \
        --eval_dir=path/to/eval_dir \
        --eval_config_path=eval_config.pbtxt \
        --model_config_path=model_config.pbtxt \
        --input_config_path=eval_input_config.pbtxt
"""
import functools
import os
import tensorflow as tf
from tensorflow.python.util.deprecation import deprecated
from object_detection.builders import dataset_builder
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.legacy import evaluator
from object_detection.utils import config_util
from object_detection.utils import label_map_util

import absl.logging as logging
logging.set_verbosity(logging.INFO)
tf.compat.v1.disable_eager_execution()
tf.config.experimental_run_functions_eagerly(True)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', type=str, help='Directory where the checkpoint files are stored')
parser.add_argument('--eval_dir', type=str, help='Directory where the eval_dir files are stored')
parser.add_argument('--pipeline_config_path', type=str, help='Directory where the pipeline_config_path files are stored')
parser.add_argument('--model_config_path', type=str, help='Directory where the model files are stored')

parser.add_argument('--eval_config_path', type=str, help='Directory where the model files are stored')
parser.add_argument('--input_config_path', type=str, help='Directory where the model files are stored')

parser.add_argument('--eval_training_data', type=bool)
parser.add_argument('--run_once', type=bool , default=True)

args = parser.parse_args()



@deprecated(None, 'Use object_detection/model_main.py.')
def main(unused_argv):
  assert args.checkpoint_dir, '`checkpoint_dir` is missing.'
  assert args.eval_dir, '`eval_dir` is missing.'
  tf.io.gfile.makedirs(args.eval_dir)
  if args.pipeline_config_path:
    configs = config_util.get_configs_from_pipeline_file(
        args.pipeline_config_path)
    tf.io.gfile.copy(
        args.pipeline_config_path,
        os.path.join(args.eval_dir, 'pipeline.config'),
        overwrite=True)
  else:
    configs = config_util.get_configs_from_multiple_files(
        model_config_path=args.model_config_path,
        eval_config_path=args.eval_config_path,
        eval_input_config_path=args.input_config_path)
    for name, config in [('model.config', args.model_config_path),
                         ('eval.config', args.eval_config_path),
                         ('input.config', args.input_config_path)]:
      tf.io.gfile.copy(config, os.path.join(args.eval_dir, name), overwrite=True)

  model_config = configs['model']
  eval_config = configs['eval_config']
  input_config = configs['eval_input_config']
  if args.eval_training_data:
    input_config = configs['train_input_config']

  model_fn = functools.partial(
      model_builder.build, model_config=model_config, is_training=False)
  @tf.function
  def get_next(config):
    dataset = dataset_builder.build(config)
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    return iterator.get_next()



  create_input_dict_fn = functools.partial(get_next, input_config)

  categories = label_map_util.create_categories_from_labelmap(
      input_config.label_map_path)

  if args.run_once:
    eval_config.max_evals = 1

  graph_rewriter_fn = None
  if 'graph_rewriter_config' in configs:
    graph_rewriter_fn = graph_rewriter_builder.build(
        configs['graph_rewriter_config'], is_training=False)

  evaluator.evaluate(
      create_input_dict_fn,
      model_fn,
      eval_config,
      categories,
      args.checkpoint_dir,
      args.eval_dir,
      graph_hook_fn=graph_rewriter_fn)


if __name__ == '__main__':
  tf.compat.v1.app.run()