'''
Based on Tensorflow's boston.py tutorial at 
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/input_fn/boston.py

'''
import numpy as np
import tensorflow as tf
import itertools
import pandas as pd
import os
import json
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec

import utils

#all input and output nodes
#COLUMNS = ["time", "tiltx", "tilty", "tilty", "wifibssid", "wifilevel"]
COLUMNS = ["time", "usageEventsName", "auth"]
#input
FEATURES = ["time", "usageEventsName"]
#output
LABEL = "auth"

#directory that stores parameters and saves model after training
MODEL_DIR = "bms1_model"

#set to train on, has input and output data
TRAINING_SET = "58.json"
#set to test on, has input and output data
TEST_SET = "58.json"
#set to predict output for, may not have output data
PREDICTION_SET = "58.json"

#converts string to int
def stringToBits(str, numBits):
    h = 13
    length = len(str)
    for i in range(length):
        h = 31*h + ord(str[i])
    return h

def input_fn(data_set):

  time_column_data = []
  auth_column_data = []
  usageEvents_column_data = []

  script_dir = os.path.dirname(__file__)
  file_path = os.path.join(script_dir, data_set)
  with open(file_path) as data_file:    
    data = json.load(data_file)
  num_examples = len(data)
  print('len: ', num_examples)

  for index in range(num_examples):
    if('usageEvents' in data[index]):
      print('usage events found')
    time_column_data.append(len(data[index]['time']))
    usageEvents_column_data.append(1)
    auth_column_data.append(1)
    
  time_column_data = np.asarray(time_column_data)
  auth_column_data = np.asarray(auth_column_data)
  usageEvents_column_data = np.asarray(usageEvents_column_data)

  time_tensor = tf.constant(time_column_data)
  usageEvents_tensor = tf.constant(usageEvents_column_data)
  labels = tf.constant(auth_column_data)


  feature_columns = {}

  feature_columns["time"] = time_tensor
  feature_columns["usageEventsName"] = usageEvents_tensor

  return feature_columns, labels


def main():
  print('main')
  input_fn(TEST_SET)

  feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]
    # Build 2 layer fully connected DNN with 10, 10 units respectively.
  regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                            hidden_units=[8, 8],
                                            model_dir=MODEL_DIR)
  #training on training set
  regressor.fit(input_fn=lambda: input_fn(TRAINING_SET), steps=2)
  #evaluate on testting set
  ev = regressor.evaluate(input_fn=lambda: input_fn(TEST_SET), steps=1)
  loss_score = ev["loss"]
  print("Loss: {0:f}".format(loss_score))

  # Print out predictions on the prediction set
  y = regressor.predict(input_fn=lambda:input_fn(PREDICTION_SET))
  # .predict() returns an iterator; convert to a list and print predictions
  predictions = list(itertools.islice(y, 6))
  print("Predictions: {}".format(str(predictions)))

if __name__== "__main__":
  main()



