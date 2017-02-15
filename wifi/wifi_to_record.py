#based off of tensorflow's convert_to_records.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import json

import tensorflow as tf


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

def stringToBits(str, numBits):
    h = 13
    length = len(str)
    for i in range(length):
        h = 31*h + ord(str[i])
    return h


def main(unused_argv):
    print("wifi_to_record:main")
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'wifi_test.json')
    with open(file_path) as data_file:    
        data = json.load(data_file)




    num_examples = len(data)

    name = 'wifi_record'
    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        print(data)

        time = data[index]['time']
        time = stringToBits(time, 4)
        time1 = time%2
        
        

        wifis = data[index]['wifi']

        numWifis = len(wifis)
        for w in range(numWifis):
            wifi = wifis[w]
            ssid = stringToBits(wifi['ssid'], 4)
            ssid1 = ssid%2
            level = stringToBits(wifi['level'], 4)
            level1 = level%2

            example = tf.train.Example(features=tf.train.Features(feature={
                'time1': _float_feature(time1),
                'ssid1': _float_feature(ssid1),
                'level1': _float_feature(level1)
                }))
            writer.write(example.SerializeToString())
    writer.close()
    




if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--directory',
      type=str,
      default='.',
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
