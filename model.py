from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

import tensorflow as tf




def make_model():
  # (/device:GPU:0
  with tf.device("/device:GPU:0"):
    LABEL_DIMENSIONS = 10
    inputs = tf.keras.Input(shape=(28,28,1))  
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    predictions = tf.keras.layers.Dense(LABEL_DIMENSIONS, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.summary()
    return model




 
def _my_model_fn(features, labels, mode):
  # with tf.device(...): # You can set device if using GPUs
   
  # define network and inference
  # (simple 2 fully connected hidden layer : 784->128->64->10)
  sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

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
 