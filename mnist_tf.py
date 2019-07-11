from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
  
import sys
import argparse
import math
import json
import tensorflow as tf
from ngrok_custom import run_with_ngrok
import os
from requests import get
from model import *
from data import *
from keras.datasets import mnist

FLAGS = None
batch_size = 100


def main(_):
  if(len(FLAGS.NAT) >= 1):
    ip = get('https://api.ipify.org').text
    print('Public IP address is: {}'.format(ip))
  else:  
    run_with_ngrok(FLAGS.port, FLAGS.token)
  parameter_nodes = []
  chief_nodes = []
  worker_nodes = []
  for i in range(FLAGS.n_parameter):
    text = input(f'Host of parameter node {i}: ')
    parameter_nodes.append(text)
  for i in range(FLAGS.n_chief):
    text = input(f'Host of chief node {i}: ')
    chief_nodes.append(text)
  for i in range(FLAGS.n_worker):
    text = input(f'Host of worker node {i}: ')
    worker_nodes.append(text)

  if FLAGS.type =='worker':
    FLAGS.index = 1
  else:
    FLAGS.index = 0
  cluster = {}
  if len(chief_nodes) > 0:
    cluster['chief'] = chief_nodes
  if len(parameter_nodes) > 0:
    cluster['ps'] = parameter_nodes
  if len(worker_nodes) > 0:
    cluster['worker'] = worker_nodes
  # cluster = {'chief': chief_nodes,
  #            'ps': parameter_nodes,
  #            'worker': worker_nodes}
  os.environ['TF_CONFIG'] = json.dumps(
      {'cluster': cluster,
       'task': {'type': FLAGS.type, 'index': FLAGS.task_id}})
  # read TF_CONFIG
  strategy = None
  if FLAGS.type == 'worker' and len(worker_nodes) >= 2:
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    print(strategy)
  run_config = tf.estimator.RunConfig(train_distribute=strategy)
  # run_config = tf.estimator.RunConfig()
  # 'chief', 'evaluator', 'master', 'ps', 'worker'
  # define

  model = make_model()
  (X_train, y_train), (X_test, y_test) = mnist.load_data()

  estimator = tf.keras.estimator.model_to_estimator(
    keras_model = model,
    model_dir=FLAGS.out_dir,
    config=run_config
  )

  # mnist_fullyconnected_classifier = tf.estimator.Estimator(
  #   model_fn=_my_model_fn,
  #   model_dir=FLAGS.out_dir,
  #   config=run_config)
  train_spec = tf.estimator.TrainSpec(
    input_fn=_get_input_fn(X_train, y_train, 16),
    max_steps=60000 * 2 / batch_size)
  eval_spec = tf.estimator.EvalSpec(
    input_fn=_get_input_fn(X_train, y_train, 8),
    steps=10000 * 1 / batch_size,
    start_delay_secs=0)
     
  # run !
  tf.estimator.train_and_evaluate(
    mnist_fullyconnected_classifier,
    train_spec,
    eval_spec
  )
              
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--train_file',
    type=str,
    default='./train.tfrecords',
    help='File path for the training data.')
  parser.add_argument(
    '--test_file',
    type=str,
    default='./test.tfrecords',
    help='File path for the test data.')
  parser.add_argument(
    '--out_dir',
    type=str,
    default='./models',
    help='Dir path for the model and checkpoint output.')
  parser.add_argument(
    '--n_parameter',
    required=True,
    type=int,
    default=1,
    help='Number ps')
  parser.add_argument(
    '--n_chief',
    required=True,
    type=int,
    default=1,
    help='Number chief')
  parser.add_argument(
    '--n_worker',
    required=True,
    type=int,
    default=1,
    help='Number worker')
  parser.add_argument(
    '--type',
    required=True,
    type=str,
    default=1,
    help='Type are worker, chief, ps, evaluator')
  parser.add_argument(
    '--port',
    required=True,
    type=str,
    default=1,
    help='Port number')
  parser.add_argument(
    '--task_id',
    required=True,
    type=int,
    default=1,
    help='Task id')
  parser.add_argument(
    '--token',
    required=False,
    type=str,
    default=1,
    help='token ngrok')
  parser.add_argument(
    '--NAT',
    required=False,
    type=str,
    default="",
    help='token ngrok')
  FLAGS, unparsed = parser.parse_known_args()

  
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
