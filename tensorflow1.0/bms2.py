'''
Based on Tensorflow's boston.py tutorial at 
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/input_fn/boston.py

'''
import numpy as np
import tensorflow as tf
import itertools
import os
import json
# from pickle import Unpickler
import utils

#all input and output nodes
#COLUMNS = ["time", "tiltx", "tilty", "tilty", "wifibssid", "wifilevel"]
COLUMNS = ["auth"]
#input
FEATURES = []
#output
LABEL = "auth"

lastFeatures = {}
sensors = {}

#directory that stores parameters and saves model after training
MODEL_DIR = "bms2_model"

file = "22.json"

#set to train on, has input and output data
TRAINING_SET = file
#set to test on, has input and output data
TEST_SET = file
#set to predict output for, may not have output data
PREDICTION_SET = file

#sensor class for wifi, appusagestats,...


class sensor:

  def __init__(self, name):
    self.name = name
    self.sensorFeatures = []

  def process(self, sensorData):
    print(self.name)
    print('sensor: process: sensordata: %s' %sensorData)
    print(len(self.sensorFeatures))
    print('processing sensor: %s' % self.name)
    for sensorFeature in self.sensorFeatures:
      sensorFeature.process(sensorData)

  def addFeature(self, sensorFeature):
    self.sensorFeatures.append(sensorFeature)

#feature within a sensor, ie wifiname, appname,...


class sensorFeature:
  #featureName  - name of feature, used as dictionary key
  #processes sensor data and updates lastFeatures with new data
  #sensorData - json object for the sensor, ie {time: t, wifi: []}

  def process(self, sensorData):
    print('processing feature: %s' % self.name)

#adds a sensor
#sensor - sensor to addSensor
def addSensor(sensor):
  sensors[sensor.name] = sensor
  print sensors
#adds a feature
#feature - feature to addSensor
#sensor - sensor the feature belongs to


def addFeature(feature, sensor):
  #initialize feature value with 1
  lastFeatures[feature.name] = 1
  sensor.addFeature(feature)
  COLUMNS.append(feature.name)
  FEATURES.append(feature.name)


#TILT
tiltSensor = sensor('tilt') #Tilt sensor exists
addSensor(tiltSensor) #adds to a global array of sensors

class TiltXFeature(sensorFeature):
  name = 'tiltX'
  def process(self, sensorData):
    lastFeatures[self.name] = sensorData['tilt'][0]
tiltXFeature = TiltXFeature() #instantiates TiltXFeature
addFeature(tiltXFeature, tiltSensor) #adds the X feature to the array inside the sensor

#USAGE EVENTS
usageEventsSensor = sensor('usageEvents') #usage events sensor exists
addSensor(usageEventsSensor) #add to a global array of sensors

class UsageEventsNameFeature(sensorFeature):
  name = 'usageEventsName'
  def process(self, sensorData):
    # lastFeatures['usageEventsName'] = len(sensorData['usageEvents'][0]['name'])
    lastFeatures['usageEventsName'] = stringToBits(sensorData['usageEvents'][0]['name'], 1)
usageEventsNameFeature = UsageEventsNameFeature()
addFeature(usageEventsNameFeature, usageEventsSensor)

#USAGE STATISTICS
usageStatsSensor = sensor('usageStats')
addSensor(usageStatsSensor)

class UsageStatsNameFeature(sensorFeature):
  name = 'usageStatsName'
  def process(self, sensorData):
    lastFeatures['usageStatsName'] = len(sensorData['usageStats'][0]['name'])
usageStatsNameFeature = UsageStatsNameFeature()
addFeature(usageStatsNameFeature, usageStatsSensor)

#WIFI
wifiSensor = sensor('wifi')
addSensor(wifiSensor)

class WifiLevelFeature(sensorFeature):
  name = 'wifiLevel'
  def process(self, sensorData):
    lastFeatures['wifiLevel'] = len(sensorData['wifi'][0]['level'])
wifiLevelFeature = WifiLevelFeature()
addFeature(wifiLevelFeature, wifiSensor)

#LOCATION
locationSensor = sensor('location')
addSensor(locationSensor)

#converts string to int
def stringToBits(str, numBits):
  h = 13
  length = len(str)
  for i in range(length):
    h = 31 * h + ord(str[i])
  return h % 4294967296


def input_fn(data_set):

  auth_column_data = []
  usageEventsName_column_data = []

  #dictionary for feature data, key is feature name and value is list of data_set
  #all list should be same size
  #one neural network input will be one number from each list
  column_data = {}
  for key in FEATURES:
    column_data[key] = []

  script_dir = os.path.dirname(__file__)
  file_path = os.path.join(script_dir, data_set)
  with open(file_path) as data_file:        
    data = json.load(data_file)
  num_examples = len(data)

  for index in range(num_examples):
    print('index: %s' % index)
    emptyData = False
    for key in sensors:
      if key in data[index]:
        # if(len(data[index][key])==0):
        #   print('empty sensor read in')
        #   emptyData = True
        #   break
        print('matching key found: %s' % key)
        sensors[key].process(data[index])
        break
    if emptyData:
      continue
    auth_column_data.append(1)
    for key in column_data:
      column_data[key].append(lastFeatures[key])
    #usageEventsName_column_data.append(len(lastFeatures['usageEventsName']))

  auth_column_data = np.asarray(auth_column_data)
  for key in column_data:
      column_data[key] = np.asarray(column_data[key])
  # usageEventsName_column_data = np.asarray(usageEventsName_column_data)

  tensors = {}
  for key in column_data:
      tensors[key] = tf.constant(column_data[key])

  #usageEventsName_tensor = tf.constant(usageEventsName_column_data)
  labels = tf.constant(auth_column_data)

  feature_columns = {}
  for key in column_data:
      feature_columns[key] = tensors[key]
  # feature_columns["usageEventsName"] = usageEventsName_tensor

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
  y = regressor.predict(input_fn=lambda: input_fn(PREDICTION_SET))
  # .predict() returns an iterator; convert to a list and print predictions
  predictions = list(itertools.islice(y, 6))
  print("Predictions: {}".format(str(predictions)))


if __name__ == "__main__":
  main()
