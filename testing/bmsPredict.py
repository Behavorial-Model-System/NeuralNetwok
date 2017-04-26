import os
import json
import sys
import bms4

'''
Runs bms4.py prediction on json files in given folder, starting at the end
and stopping when the given fraction of files is completed
'''

def main(argv):
  if(len(argv)!=3):
    print('usage: bmsPredict.py <folder filepath> <fraction from end to predict> <results filepath')
    print('usage example: bmsPredict.py 86768602162866Data 0.25 prediction_results.txt')
    return
  folderPath = argv[0]
  fraction = float(argv[1])
  resultsPath = argv[2]

  filePathList = []

  for filename in os.listdir(folderPath):
    filePathList.append(folderPath + '/' + filename)
  filePathList.sort()

  numberToPredict = int(len(filePathList) * fraction)
  total = 0
  count = 0
  #opening the file
  target = open(resultsPath, 'w')
  predictionList = []

  for i in range(len(filePathList) - numberToPredict, len(filePathList)):
    print('progress: %f' %(float(count)/numberToPredict))
    prediction = bms4.main(['predict', filePathList[i]])
    predictionList.append(prediction)
    total += prediction
    count += 1
  average = total / (numberToPredict)

  target.write('average prediction over {0} json files: {1}\n'.format(count, average))
  for i in range(0, len(predictionList)):
    target.write('{0} , {1}\n'.format(filePathList[i], predictionList[i]))
  target.close()
  print('prediction: %f\n' %average)




if __name__ == "__main__":
  main(sys.argv[1:])