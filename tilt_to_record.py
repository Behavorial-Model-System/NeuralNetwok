#based off of tensorflow's convert_to_records.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import json

import tensorflow as tf

#from tensorflow.contrib.learn.python.learn.datasets import mnist

FLAGS = None


#feature for integers
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#feature for floats
def _float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value= [value]))
#feature for strings and others
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# class Timestamp:
#     int month, day, year
#     int hour, minute, second
#     def __init__(self, String)

# def class TiltRecord:
#     float x, y, z


def convert_to(data_set, name):
    print("tilt_to_record:convert_to")


def main(unused_argv):
    print("tilt_to_record:main")
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'tilt_test.json')
    with open(file_path) as data_file:    
        data = json.load(data_file)

    print(data)
    print("\n\n")
    print(data[0]['time'])

    num_examples = 2

    name = 'tilt_test'


    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        example = tf.train.Example(features=tf.train.Features(feature={
            'time': _bytes_feature(str(data[index]['time'])),
            'tiltx': _float_feature(data[index]['tilt'][0]),
            'tilty': _float_feature(data[index]['tilt'][1]),
            'tiltz': _float_feature(data[index]['tilt'][2]),
            
            }))
        writer.write(example.SerializeToString())
    writer.close()

    

    
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--directory',
      type=str,
      default='/tmp/data',
      help='Directory to download data files and write the converted result'
  )
  parser.add_argument(
      '--validation_size',
      type=int,
      default=5000,
      help="""\
      Number of examples to separate from the training data for the validation
      set.\
      """
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
