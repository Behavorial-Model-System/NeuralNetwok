#from bms3.py import main
import bms3
import os
import json
import glob

print('running analysisScript..........')


folder = '86768602162866Data'#
#bms3.main(['train', file, 1])

filelist = []


for filename in os.listdir(folder):
  filelist.append(folder + '/'+ filename)
filelist.sort()



numberToTrain = len(filelist)*3/4

for i in range(0, numberToTrain):
  print(float(i) / numberToTrain)
  bms3.main(['train', filelist[i], 1])


'''
numberToPredict = len(filelist)*1/4
total = 0
for i in range(numberToTrain, len(filelist)):
  print(float(i) - numberToTrain) / (len(filelist) - numberToTrain)
  prediction = bms3.main(['predict', filelist[i]])
  total+= prediction
average = total / (len(filelist)-numberToTrain)
print('average = %f' % average)
'''
'''
for filename in os.listdir(folder):
  with open(folder + '/'+ filename) as data_file:
    print(filename)
    data = json.load(data_file)
'''