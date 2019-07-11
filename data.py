# from keras.datasets import mnist
import tensorflow as tf
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train = X_train.reshape(60000,28,28,1)
# X_test = X_test.reshape(10000,28,28,1)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)


# def input_fn():
#     datasets, ds_info = tfds.load(name='mnist', with_info=True, as_supervised=True)
#     mnist_train, mnist_test = datasets['train'], datasets['test']

#     BUFFER_SIZE = 10000
#     BATCH_SIZE = 64

#     def scale(image, label):
#         image = tf.cast(image, tf.float32)
#         image /= 255
    
#         return image, label[..., tf.newaxis]

#     train_data = mnist_train.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
#     return train_data.repeat()



# def _my_input_fn(filepath, num_epochs):
#   # image - 784 (=28 x 28) elements of grey-scaled integer value [0, 1]
#   # label - digit (0, 1, ..., 9)
#   data_queue = tf.train.string_input_producer(
#     [filepath],
#     num_epochs = num_epochs) # data is repeated and it raises OutOfRange when data is over
#   data_reader = tf.TFRecordReader()
#   _, serialized_exam = data_reader.read(data_queue)
#   data_exam = tf.parse_single_example(
#     serialized_exam,
#     features={
#       'image_raw': tf.FixedLenFeature([], tf.string),
#       'label': tf.FixedLenFeature([], tf.int64)
#     })
#   data_image = tf.decode_raw(data_exam['image_raw'], tf.uint8)
#   data_image.set_shape([784])
#   data_image = tf.cast(data_image, tf.float32) * (1. / 255)
#   data_label = tf.cast(data_exam['label'], tf.int32)
#   data_batch_image, data_batch_label = tf.train.batch(
#     [data_image, data_label],
#     batch_size=batch_size)
#   return data_batch_image, data_batch_label

def _my_input_fn(features, labels, batch_size):
  dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
  return dataset.shuffle(1000).repeat().batch(batch_size)
 
def _get_input_fn(features, labels, batch_size):
  return lambda: _my_input_fn(filepath, num_epochs)