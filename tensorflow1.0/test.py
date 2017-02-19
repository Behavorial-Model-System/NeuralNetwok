import numpy as np
import tensorflow as tf
import os
import json
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec

LEARNING_RATE = 0.001
INPUT_UNITS = 2
HIDDEN_UNITS = 3
OUTPUT_UNITS = 1

# Declare list of features, we only have one real-valued feature
def model(features, labels, mode):

  print("features['x']", features['x'])


  inputs = tf.placeholder(tf.float32, shape=[None, 2])
  desired_outputs = tf.placeholder(tf.float32, shape=[None, 1])

  # Build a linear model and predict values
  W1 = tf.get_variable("W1", [HIDDEN_UNITS, INPUT_UNITS], dtype=tf.float64)
  b1 = tf.get_variable("b1", [HIDDEN_UNITS], dtype=tf.float64)
  
  W1x1 =  W1*features['x']
  print('x', features['x'].shape)
  h1 = W1x1+ b1

  W2 = tf.get_variable("W2", [OUTPUT_UNITS, HIDDEN_UNITS], dtype=tf.float64)
  b2 = tf.get_variable("b2", [OUTPUT_UNITS], dtype=tf.float64)
  y = W2* h1 + b2
  
  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  # ModelFnOps connects subgraphs we built to the
  # appropriate functionality.
  return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=y,
      loss= loss,
      train_op=train)

estimator = tf.contrib.learn.Estimator(model_fn=model)
# define our data set
'''
x=np.array([1., 2., 3., 4.])
y=np.array([0., -1., -2., -3.])
'''

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'food_test.json')
with open(file_path) as data_file:    
  data = json.load(data_file)
num_examples = len(data)
x1 = []
x2 = []
x = []
y = []
for index in range(num_examples):
  x.append([data[index]['food'], data[index]['water']])
  x1.append(data[index]['food'])
  x2.append(data[index]['water'])
  y.append(data[index]['happiness'])
x1 = np.asarray(x1)
x2 = np.asarray(x2)
y = np.asarray(y)

x = np.asarray(x)
print('x', x)

'''
validation_metrics = {

  "accuracy":
    tf.contrib.learn.MetricSpec(
    metric_fn=tf.contrib.metrics.streaming_accuracy,
    prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
  "precision":
    tf.contrib.learn.MetricSpec(
    metric_fn=tf.contrib.metrics.streaming_precision,
    prediction_key=tf.contrib.learn.PredictionKey.
    CLASSES),
  "recall":
    tf.contrib.learn.MetricSpec(
    metric_fn=tf.contrib.metrics.streaming_recall,
    prediction_key=tf.contrib.learn.PredictionKey.
    CLASSES)

}

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
  x,
  y,
  every_n_steps=50,
  metrics=validation_metrics,
  early_stopping_metric="loss",
  early_stopping_metric_minimize=True,
  early_stopping_rounds=200)
  '''
# input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x1 }, y, 1, num_epochs=1000)
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x }, y, 1, num_epochs=1000)
'''
# train
estimator.fit(input_fn=input_fn, steps=1000)
# evaluate our model
print(estimator.evaluate(input_fn=input_fn, steps=10))
'''

for _ in range(10):
  # train
  estimator.partial_fit(input_fn=input_fn, steps=10)
  # evaluate our model
  print(estimator.evaluate(input_fn=input_fn, steps=10))