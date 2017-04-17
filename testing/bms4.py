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

class NetworkSetup:
  #all input and output nodes
  #COLUMNS = ["auth", "time", "tiltx", "tilty", "tilty", "wifibssid", "wifilevel"]
  COLUMNS = ["auth"]
  #input
  FEATURES = []
  #output
  LABEL = "auth"

  lastFeatures = {}
  column_data = {}
  sensors = {}


  #directory that stores parameters and saves model after training
  MODEL_DIR = "bms4_model"
  lastFeaturesPickle = 'lastFeatures.pckl'

  def addSensor(self, sensor):
    self.sensors[sensor.name] = sensor
    self.numTrainingPoints = 0

  def addFeature(self, featureName):
    self.FEATURES.append(featureName)
    self.lastFeatures[featureName] = 0
    self.column_data[featureName] = []

  def feedLastFeatures(self):
    for key in self.FEATURES:
      self.column_data[key].append(self.lastFeatures[key])
    self.numTrainingPoints += 1

networkSetup = NetworkSetup()

#sensor class for wifi, appusagestats,...
class Sensor:
  def __init__(self, name):
    self.name = name

  def process(self, sensorData):
    print('must be implemented')

# GENERAL
class GeneralSensor(Sensor):
  def __init__(self):
    self.name = 'general'
    networkSetup.addSensor(self)
    networkSetup.addFeature('hour')
    networkSetup.addFeature('minute')
  def process(self, sensorData):
    hour, minute = stampToTimeUnits(sensorData['time'])
    networkSetup.lastFeatures['hour'] = hour
    networkSetup.lastFeatures['minute'] = minute
    networkSetup.feedLastFeatures()
generalSensor = GeneralSensor()


#TILT
class TiltSensor(Sensor):
  def __init__(self):
    self.name = 'tilt'
    networkSetup.addSensor(self)
    networkSetup.addFeature('tiltX')
    networkSetup.addFeature('tiltY')
    networkSetup.addFeature('tiltZ')
  def process(self, sensorData):
    print('tilt processing')
    networkSetup.lastFeatures['tiltX'] = sensorData['tilt'][0]
    networkSetup.lastFeatures['tiltY'] = sensorData['tilt'][1]
    networkSetup.lastFeatures['tiltZ'] = sensorData['tilt'][2]
    networkSetup.feedLastFeatures()
tiltSensor = TiltSensor()

#USAGE EVENTS
class AppUsageEventsSensor(Sensor):
  def __init__(self):
    self.name = 'usageEvents'
    networkSetup.addSensor(self)
    networkSetup.addFeature('usageEventsName')
    networkSetup.addFeature('usageEventsType')
  def process(self, sensorData):
    print('usage events processing')
    numEvents = len(sensorData['usageEvents'])
    for i in range(numEvents):
      networkSetup.lastFeatures['usageEventsName'] = len(sensorData['usageEvents'][i]['name'])
      networkSetup.lastFeatures['usageEventsType'] = sensorData['usageEvents'][i]['type']
      networkSetup.feedLastFeatures()
appUsageEventsSensor = AppUsageEventsSensor()

#WIFI
class WifiSensor(Sensor):
  def __init__(self):
    self.name = 'wifi'
    networkSetup.addSensor(self)
    networkSetup.addFeature('wifiBssid')
    networkSetup.addFeature('wifiLevel')
  def process(self, sensorData):
    print('wifi processing')
    numWifis = len(sensorData['wifi'])
    for i in range(numWifis):
      networkSetup.lastFeatures['wifiBssid'] = len(sensorData['wifi'][i]['bssid'])
      networkSetup.lastFeatures['wifiLevel'] = sensorData['wifi'][i]['level']
      networkSetup.feedLastFeatures()
wifiSensor = WifiSensor()

#LOCATION
class LocationSensor(Sensor):
  def __init__(self):
    self.name = 'location'
    networkSetup.addSensor(self)
    networkSetup.addFeature('longitude')
    networkSetup.addFeature('latitude')
  def process(self, sensorData):
    print('location processing')
    networkSetup.lastFeatures['longitude'] = sensorData['location']['longitude']
    networkSetup.lastFeatures['latitude'] = sensorData['location']['latitude']
    networkSetup.feedLastFeatures()
locationSensor = LocationSensor()


def pickleLoad(filename):
  try:
    f = open(filename, 'r')
    networkSetup.lastFeatures = pickle.load(f)
    f.close()
  except(IOError):
    print('lastFeatures.pckl not found, using zeroes as default')

def pickleSave(filename):
  # save the last features in a pickle
  f = open(filename, 'w')
  pickle.dump(networkSetup.lastFeatures, f)
  f.close()

#converts string to int
def stringToInt(str):
  h = 13
  length = len(str)
  for i in range(length):
    h = 31 * h + ord(str[i])
  return h % 4294967296

# "time": "09-04-2017 01:26:27"
def stampToTimeUnits(str):
  hour = int(str[11:13])
  minute = int(str[14:16])
  #second = int(str[17:19])
  return hour, minute


def input_fn(data_set, isAuthentic = 1):
  auth_column_data = []

  script_dir = os.path.dirname(__file__)
  file_path = os.path.join(script_dir, data_set)
  with open(file_path) as data_file:        
    data = json.load(data_file)
  numSensorData = len(data)

  for index in range(numSensorData):
    #print('index: %s' % index)
    emptyData = True
    for key in networkSetup.sensors:
      if key in data[index]:
        #print('matching key found: %s' % key)
        networkSetup.sensors[key].process(data[index])
        networkSetup.sensors['general'].process(data[index])
        emptyData = False
        break
    if emptyData:
      continue


  for _ in range(networkSetup.numTrainingPoints):
    auth_column_data.append(isAuthentic)
  column_data = networkSetup.column_data
  column_data_np = {}

  auth_column_data = np.asarray(auth_column_data)
  for key in column_data:
    column_data_np[key] = np.asarray(column_data[key])
  # usageEventsName_column_data = np.asarray(usageEventsName_column_data)

  tensors = {}
  for key in column_data_np:
      tensors[key] = tf.constant(column_data_np[key])

  #usageEventsName_tensor = tf.constant(usageEventsName_column_data)
  labels = tf.constant(auth_column_data)

  feature_columns = {}
  for key in column_data_np:
      feature_columns[key] = tensors[key]
  # feature_columns["usageEventsName"] = usageEventsName_tensor

  return feature_columns, labels

def getRegressor():
  feature_cols = [tf.contrib.layers.real_valued_column(k) for k in networkSetup.FEATURES]
  # Build 2 layer fully connected DNN with 8, 8 units respectively.
  regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                            hidden_units=[8, 8],
                                            model_dir=networkSetup.MODEL_DIR,
                                            activation_fn=tf.nn.sigmoid,
                                            optimizer=tf.train.GradientDescentOptimizer(
                                              #learning_rate=0.001
                                              learning_rate=0.1
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
  predictions = list(itertools.islice(y, networkSetup.numTrainingPoints))
  #print("Predictions: {}".format(str(predictions)))
  sum = 0
  for p in predictions:
    sum+= p
  average = float(sum)/len(predictions)
  print('average prediction over sensor objects: %f' %(average))
  return average

def printUsage():
  print('Usage: python bms2.py <mode> <filepath> <isAuthentic>')
  print('Example: python bms2.py train 22.json 1 ')
  print('Example: python bms2.py evaluate 22.json 0 ')
  print('Example: python bms2.py predict 22.json ')
def main(argv):
  print('main')
  #input_fn('22a.json')

  if len(argv) < 2:
    printUsage()
    return
  mode = argv[0]
  filepath = argv[1]
  isAuthentic = None
  if(argv>2):
    isAuthentic = int(argv[2])
  if(mode=='train'):
    pickleLoad(networkSetup.lastFeaturesPickle)
    train(filepath, isAuthentic)
    pickleSave(networkSetup.lastFeaturesPickle)
  elif(mode == 'evaluate'):
    pickleLoad(networkSetup.lastFeaturesPickle)
    evaluate(filepath, isAuthentic)
  elif(mode == 'predict'):
    pickleLoad(networkSetup.lastFeaturesPickle)
    average = predict(filepath)
    return average
  else:
    print('first arguemnt was not "train", "evaluate", or "predict" ')






if __name__ == "__main__":
  #main(['train', '22a.json', '1'])
  main(sys.argv[1:])