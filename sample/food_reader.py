#based off of tensorflow's fully_connected_reader

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time

import tensorflow as tf


# Basic model parameters as external flags.
FLAGS = None

# Constants used for dealing with the files
TRAIN_FILE = 'food_record.tfrecords'
# For simple testing purposes, use training file for validation 
VALIDATION_FILE = 'food_record.tfrecords'


def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'food': tf.FixedLenFeature([], tf.float32),
          'happiness': tf.FixedLenFeature([], tf.float32)
      })
  
  
  food = tf.cast(features['food'], tf.float32)
  happiness = tf.cast(features['happiness'], tf.float32)


  food = tf.expand_dims(food, -1)

  print("food shape: ", tf.shape(food))
  print("happiness shape: ", tf.shape(happiness))

  return food, happiness


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
    food, happiness = read_and_decode(filename_queue)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    foods, happinesses= tf.train.shuffle_batch(
        [food, happiness], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)

    return foods, happinesses





def main(_):
  with tf.Graph().as_default():
    # Input images and labels.
    foods, happinesses = inputs(train=True, batch_size=FLAGS.batch_size,
                            num_epochs=FLAGS.num_epochs)

    HIDDEN_UNITS = 4 

    INPUTS = 1
    OUTPUTS = 1


    weights_1 = tf.Variable(tf.truncated_normal([INPUTS, HIDDEN_UNITS]))
    biases_1 = tf.Variable(tf.zeros([HIDDEN_UNITS]))

    layer_1_outputs = tf.nn.sigmoid(tf.matmul(foods, weights_1) + biases_1)

    weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS, OUTPUTS]))
    biases_2 = tf.Variable(tf.zeros([OUTPUTS]))

    logits = tf.nn.sigmoid(tf.matmul(layer_1_outputs, weights_2) + biases_2)
    
    loss = tf.reduce_mean(logits)

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
      default='.',
      help='Directory with the training data.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

