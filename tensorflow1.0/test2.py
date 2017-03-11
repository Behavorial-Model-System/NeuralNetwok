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

#all input and output nodes
COLUMNS = ["food", "water", "happiness"]
#input
FEATURES = ["food", "water"]
#output
LABEL = "happiness"

#directory that stores parameters and saves model after training
MODEL_DIR = "test2_model"

#set to train on, has input and output data
TRAINING_SET = "food_test.json"
#set to test on, has input and output data
TEST_SET = "food_test.json"
#set to predict output for, may not have output data
PREDICTION_SET = "food_test.json"


def input_fn(data_set):

  food_column_data = []
  water_column_data = []
  happiness_column_data = []

  script_dir = os.path.dirname(__file__)
  file_path = os.path.join(script_dir, data_set)
  with open(file_path) as data_file:    
    data = json.load(data_file)
  num_examples = len(data)

  for index in range(num_examples):
    food_column_data.append(data[index]['food'])
    water_column_data.append(data[index]['water'])
    happiness_column_data.append(data[index]['happiness'])
  food_column_data = np.asarray(food_column_data)
  water_column_data = np.asarray(water_column_data)
  happiness_column_data = np.asarray(happiness_column_data)

  food_tensor = tf.constant(food_column_data)
  water_tensor = tf.constant(water_column_data)
  labels = tf.constant(happiness_column_data)

  feature_columns = {}

  feature_columns["food"] = food_tensor
  feature_columns["water"] = water_tensor

  return feature_columns, labels


def main():
  print('main')

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



