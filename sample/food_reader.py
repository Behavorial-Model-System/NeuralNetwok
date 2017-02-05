#based off of tensorflow's fully_connected_reader
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import mnist

# Basic model parameters as external flags.
FLAGS = None

# Constants used for dealing with the files, matches simple_to_record.
TRAIN_FILE = 'simple_test.tfrecords'
# For simple testing purposes, use training file for validation 
VALIDATION_FILE = 'simple_test.tfrecords'


def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'time': tf.FixedLenFeature([], tf.float32),
          'tiltx': tf.FixedLenFeature([], tf.float32),
          'tilty': tf.FixedLenFeature([], tf.float32),
          'tiltz': tf.FixedLenFeature([], tf.float32),

      })


  #time = tf.decode_raw(features['time'], tf.uint8)
  #time.set_shape([20])
  
  
  time = tf.cast(features['time'], tf.float32)
  tiltx = tf.cast(features['tiltx'], tf.float32)


  time = tf.expand_dims(time, -1)

  print("time shape: ", tf.shape(time))
  print("tiltx shape: ", tf.shape(tiltx))

  return time, tiltx


def inputs(train, batch_size, num_epochs):
  """Reads input data num_epochs times.

  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.

  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """
  if not num_epochs: num_epochs = None
  filename = os.path.join(FLAGS.train_dir,
                          TRAIN_FILE if train else VALIDATION_FILE)

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename
    # queue.
    time, tiltx = read_and_decode(filename_queue)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    times, tiltxs = tf.train.shuffle_batch(
        [time, tiltx], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)

    return times, tiltxs





def main(_):
  with tf.Graph().as_default():
    # Input images and labels.
    times, tiltxs = inputs(train=True, batch_size=FLAGS.batch_size,
                            num_epochs=FLAGS.num_epochs)

    print("times:", times)
    print("tiltsx:", tiltxs)
    print("shape(times)", tf.shape(times))
    print("shape(tiltxs)", tf.shape(tiltxs))
   

    HIDDEN_UNITS = 4 

    INPUTS = 1
    OUTPUTS = 1

    #input_placeholder = tf.placeholder()

    weights_1 = tf.Variable(tf.truncated_normal([INPUTS, HIDDEN_UNITS]))
    biases_1 = tf.Variable(tf.zeros([HIDDEN_UNITS]))
    #layer_1_outputs = tf.nn.sigmoid(tf.matmul(inputs, weights_1) + biases_1)
    layer_1_outputs = tf.nn.sigmoid(tf.matmul(times, weights_1) + biases_1)

    weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS, OUTPUTS]))
    biases_2 = tf.Variable(tf.zeros([OUTPUTS]))

    logits = tf.nn.sigmoid(tf.matmul(layer_1_outputs, weights_2) + biases_2)
    
    '''
    
    #error_function = 0.5 * tf.reduce_sum(tf.sub(logits, desired_outputs) * tf.sub(logits, desired_outputs))
    error_function = 0.5 * tf.reduce_sum(tf.sub(logits, tiltxs) * tf.sub(logits, tiltxs))

    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(error_function)
    train_op = optimizer.minimize(loss)train
    
    cost = tf.nn.softmax(logits)
    '''
    loss = tf.reduce_mean(logits, 1)

    learning_rate = 0.01
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    init_op = tf.group(tf.global_variables_initializer())

    sess = tf.Session()
    sess.run(init_op)
    
    print('staring iteration', 0)
    _, loss = sess.run([train_op, loss])
    print(loss)

    sess.close()
'''
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, tiltxs, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
'''
   





  



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--num_epochs',
      type=int,
      default=2,
      help='Number of epochs to run trainer.'
  )
  parser.add_argument(
      '--hidden1',
      type=int,
      default=128,
      help='Number of units in hidden layer 1.'
  )
  parser.add_argument(
      '--hidden2',
      type=int,
      default=32,
      help='Number of units in hidden layer 2.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='Batch size.'
  )
  parser.add_argument(
      '--train_dir',
      type=str,
      default='/tmp/data',
      help='Directory with the training data.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

