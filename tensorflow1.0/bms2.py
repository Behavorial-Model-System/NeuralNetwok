'''
Based on Tensorflow's boston.py tutorial at 
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/input_fn/boston.py

'''
import numpy as np
import tensorflow as tf
import itertools
import os
import json
import pickle
import sys


#all input and output nodes
#COLUMNS = ["auth", "time", "tiltx", "tilty", "tilty", "wifibssid", "wifilevel"]
COLUMNS = ["auth"]
#input
FEATURES = []
#output
LABEL = "auth"

lastFeatures = {}
sensors = {}

#directory that stores parameters and saves model after training
MODEL_DIR = "bms2_model"
lastFeaturesPickle = 'lastFeatures.pckl'



#sensor class for wifi, appusagestats,...


class sensor:

  def __init__(self, name):
    self.name = name
    self.sensorFeatures = []

  def process(self, sensorData):
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


def pickleLoad(filename):
  try:
    f = open(filename, 'r')
    object = pickle.load(f)
    return object
    f.close()
  except(IOError):
    print('lastFeatures.pckl not found, using zeroes as default')

def pickleSave(object, filename):
  # save the last features in a pickle
  f = open(filename, 'w')
  pickle.dump(object, f)
  f.close()

#converts string to int
def stringToBits(str, numBits):
  h = 13
  length = len(str)
  for i in range(length):
    h = 31 * h + ord(str[i])
  return h % 4294967296


def input_fn(data_set, isAuthentic = 1):
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

def getRegressor():
  feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]
  # Build 2 layer fully connected DNN with 8, 8 units respectively.
  regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                            hidden_units=[8, 8],
                                            model_dir=MODEL_DIR,
                                            activation_fn=tf.nn.tanh,
                                            optimizer=tf.train.GradientDescentOptimizer(
                                              learning_rate=0.001,
                                            )
                                            )
  return regressor
def train(filepath, isAuthentic):
  regressor = getRegressor()
  # training on training set
  regressor.fit(input_fn=lambda: input_fn(filepath, isAuthentic), steps=1)

def evaluate(filepath, isAuthentic):
  regressor = getRegressor()
  # evaluate on testting set
  ev = regressor.evaluate(input_fn=lambda: input_fn(filepath), steps=1)
  loss_score = ev["loss"]
  print("Loss: {0:f}".format(loss_score))
def predict(filepath):
  regressor = getRegressor()
  y = regressor.predict(input_fn=lambda: input_fn(filepath))
  # .predict() returns an iterator; convert to a list and print predictions
  predictions = list(itertools.islice(y, 20))
  print("Predictions: {}".format(str(predictions)))

def printUsage():
  print('Usage: python bms2.py <filepath> <mode> <isAuthentic>')
  print('Example: python bms2.py train 22.json 1 ')
  print('Example: python bms2.py evaluate 22.json 0 ')
  print('Example: python bms2.py predict 22.json ')
def main(argv):
  print('main')

  if len(argv) < 2:
    printUsage()
    return
  filepath = argv[1]
  if(argv[0]=='train'):
    lastFeatures = pickleLoad(lastFeaturesPickle)
    train(filepath, argv[2])
    pickleSave(lastFeatures, lastFeaturesPickle)
  elif(argv[0] == 'evaluate'):
    lastFeatures = pickleLoad(lastFeaturesPickle)
    evaluate(filepath, argv[2])
  elif(argv[0] == 'predict'):
    lastFeatures = pickleLoad(lastFeaturesPickle)
    predict(filepath)
  else:
    print('first arguemnt was not "train", "evaluate", or "predict" ')






if __name__ == "__main__":
  main(sys.argv[1:])
