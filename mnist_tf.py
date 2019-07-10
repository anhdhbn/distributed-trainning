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
FLAGS = None
batch_size = 100
 
def _my_input_fn(filepath, num_epochs):
  # image - 784 (=28 x 28) elements of grey-scaled integer value [0, 1]
  # label - digit (0, 1, ..., 9)
  data_queue = tf.train.string_input_producer(
    [filepath],
    num_epochs = num_epochs) # data is repeated and it raises OutOfRange when data is over
  data_reader = tf.TFRecordReader()
  _, serialized_exam = data_reader.read(data_queue)
  data_exam = tf.parse_single_example(
    serialized_exam,
    features={
      'image_raw': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.int64)
    })
  data_image = tf.decode_raw(data_exam['image_raw'], tf.uint8)
  data_image.set_shape([784])
  data_image = tf.cast(data_image, tf.float32) * (1. / 255)
  data_label = tf.cast(data_exam['label'], tf.int32)
  data_batch_image, data_batch_label = tf.train.batch(
    [data_image, data_label],
    batch_size=batch_size)
  return data_batch_image, data_batch_label
 
def _get_input_fn(filepath, num_epochs):
  return lambda: _my_input_fn(filepath, num_epochs)
 
def _my_model_fn(features, labels, mode):
  # with tf.device(...): # You can set device if using GPUs
   
  # define network and inference
  # (simple 2 fully connected hidden layer : 784->128->64->10)
  with tf.name_scope('hidden1'):
    weights = tf.Variable(
      tf.truncated_normal(
        [784, 128],
        stddev=1.0 / math.sqrt(float(784))),
      name='weights')
    biases = tf.Variable(
      tf.zeros([128]),
      name='biases')
    hidden1 = tf.nn.relu(tf.matmul(features, weights) + biases)
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
      tf.truncated_normal(
        [128, 64],
        stddev=1.0 / math.sqrt(float(128))),
      name='weights')
    biases = tf.Variable(
      tf.zeros([64]),
      name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
      tf.truncated_normal(
        [64, 10],
        stddev=1.0 / math.sqrt(float(64))),
    name='weights')
    biases = tf.Variable(
      tf.zeros([10]),
      name='biases')
    logits = tf.matmul(hidden2, weights) + biases
 
  # compute evaluation matrix
  predicted_indices = tf.argmax(input=logits, axis=1)
  if mode != tf.estimator.ModeKeys.PREDICT:
    label_indices = tf.cast(labels, tf.int32) 
    accuracy = tf.metrics.accuracy(label_indices, predicted_indices)
    tf.summary.scalar('accuracy', accuracy[1]) # output to TensorBoard
 
  # compute loss
  loss = tf.losses.sparse_softmax_cross_entropy(
    labels=labels,
    logits=logits)
 
  # define operations
  if mode == tf.estimator.ModeKeys.TRAIN:
    #global_step = tf.train.create_global_step()
    #global_step = tf.contrib.framework.get_or_create_global_step()
    global_step = tf.train.get_or_create_global_step()    
    optimizer = tf.train.GradientDescentOptimizer(
      learning_rate=0.07)
    train_op = optimizer.minimize(
      loss=loss,
      global_step=global_step)
    return tf.estimator.EstimatorSpec(
      mode,
      loss=loss,
      train_op=train_op)
  if mode == tf.estimator.ModeKeys.EVAL:
    eval_metric_ops = {
      'accuracy': accuracy
    }
    return tf.estimator.EstimatorSpec(
      mode,
      loss=loss,
      eval_metric_ops=eval_metric_ops)
  if mode == tf.estimator.ModeKeys.PREDICT:
    probabilities = tf.nn.softmax(logits, name='softmax_tensor')
    predictions = {
      'classes': predicted_indices,
      'probabilities': probabilities
    }
    export_outputs = {
      'prediction': tf.estimator.export.PredictOutput(predictions)
    }
    return tf.estimator.EstimatorSpec(
      mode,
      predictions=predictions,
      export_outputs=export_outputs)
 
def main(_):
  run_with_ngrok(FLAGS.port)
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
  cluster = {'chief': chief_nodes,
             'ps': parameter_nodes,
             'worker': worker_nodes}
  os.environ['TF_CONFIG'] = json.dumps(
      {'cluster': cluster,
       'task': {'type': FLAGS.type, 'index': FLAGS.task_id}})
  # read TF_CONFIG
  run_config = tf.contrib.learn.RunConfig()
 
  # define
  mnist_fullyconnected_classifier = tf.estimator.Estimator(
    model_fn=_my_model_fn,
    model_dir=FLAGS.out_dir,
    config=run_config)
  train_spec = tf.estimator.TrainSpec(
    input_fn=_get_input_fn(FLAGS.train_file, 2),
    max_steps=60000 * 2 / batch_size)
  eval_spec = tf.estimator.EvalSpec(
    input_fn=_get_input_fn(FLAGS.test_file, 1),
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
    default='/home/demouser/train.tfrecords',
    help='File path for the training data.')
  parser.add_argument(
    '--test_file',
    type=str,
    default='/home/demouser/test.tfrecords',
    help='File path for the test data.')
  parser.add_argument(
    '--out_dir',
    type=str,
    default='/home/demouser/out',
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
    type=int,
    default=1,
    help='Port number')
  parser.add_argument(
    '--task_id',
    required=True,
    type=int,
    default=1,
    help='Task id')
  FLAGS, unparsed = parser.parse_known_args()

  
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
