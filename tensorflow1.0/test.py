import numpy as np
import tensorflow as tf
import os
import json
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec

LEARNING_RATE = 0.001
INPUT_UNITS = 2
HIDDEN_UNITS = 3
OUTPUT_UNITS = 1

COLUMNS = ["food", "water", "happiness"]
FEATURES = ["food", "water"]
LABEL = "happiness"
params = {"learning_rate": LEARNING_RATE}




def input_fn():
  food_column_data = []
  water_column_data = []
  happiness_column_data = []

  script_dir = os.path.dirname(__file__)
  file_path = os.path.join(script_dir, 'food_test.json')
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
  # feature_columns[tf.contrib.layers.real_valued_column("food")] = food_tensor
  # feature_columns[tf.contrib.layers.real_valued_column("water")] = water_tensor
  feature_columns["food"] = food_tensor
  feature_columns["water"] = water_tensor

  return feature_columns, labels


# Declare list of features, we only have one real-valued feature
def model_fn(features, labels, mode, params):


  '''
  print("features['x']", features['x'])
  inputs = tf.placeholder(tf.float32, shape=[None, 2])
  desired_outputs = tf.placeholder(tf.float32, shape=[None, 1])
  '''
  
  feature_cols = [tf.contrib.layers.real_valued_column(k) for k in features]
  input_layer = tf.contrib.layers.input_from_feature_columns(columns_to_tensors=features, feature_columns = feature_cols)
  hidden_layer = tf.contrib.layers.fully_connected(inputs=input_layer, num_outputs=10)
  output_layer = tf.contrib.layers.fully_connected(inputs=hidden_layer, num_outputs=1, activation_fn=tf.sigmoid)

  '''
  # Build a linear model and predict values
  W = tf.get_variable("W", [HIDDEN_UNITS, INPUT_UNITS], dtype=tf.float64)
  b = tf.get_variable("b", [HIDDEN_UNITS], dtype=tf.float64)
  y = W*features['x'] + b
  '''

  # Reshape output layer to 1-dim Tensor to return predictions
  predictions = tf.reshape(output_layer, [-1])
  predictions_dict = {"labels": predictions}

  # Calculate loss using mean squared error
  #loss = tf.losses.mean_squared_error(labels, predictions)
  loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(tf.cast(labels, tf.float32), predictions))))
  eval_metric_ops = {
    "rmse": tf.sqrt(tf.reduce_mean(tf.square(tf.sub(tf.cast(labels, tf.float32), predictions))))
  }

  train_op = tf.contrib.layers.optimize_loss(
    loss=loss,
    global_step=tf.contrib.framework.get_global_step(),
    learning_rate=params["learning_rate"],
    optimizer="SGD")
  
  return tf.contrib.learn.model_fn_lib.ModelFnOps(
    mode=mode,
    predictions=predictions_dict,
    loss=loss,
    train_op=train_op,
    eval_metric_ops=eval_metric_ops)


# estimator = tf.contrib.learn.Estimator(model_fn=model)

def main():
  print('main')


  
  nn = tf.contrib.learn.Estimator(model_fn = model_fn, params = params)
  # Fit
  # nn.fit(x = feature_cols, y = labels, steps=5000)
  nn.fit(input_fn = input_fn, steps=5)

  # Score accuracy
  ev = nn.evaluate(x=test_set.data, y=test_set.target, steps=1)
  print("Loss: %s" % ev["loss"])
  print("Root Mean Squared Error: %s" % ev["rmse"])

if __name__== "__main__":
  main()




# define our data set
'''
x=np.array([1., 2., 3., 4.])
y=np.array([0., -1., -2., -3.])
'''

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
# input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x }, y, 1, num_epochs=1000)
'''
# train
estimator.fit(input_fn=input_fn, steps=1000)
# evaluate our model
print(estimator.evaluate(input_fn=input_fn, steps=10))


for _ in range(10):
  # train
  estimator.partial_fit(input_fn=input_fn, steps=10)
  # evaluate our model
  print(estimator.evaluate(input_fn=input_fn, steps=10))
'''