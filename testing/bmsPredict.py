import os
import json
import sys
import bms4

'''
Runs bms4.py prediction on json files in given folder, starting at the end
and stopping when the given fraction of files is completed
'''

def main(argv):
  if(len(argv)!=2):
    print('usage: predictFolder.py <folder filepath> <fraction to train>')
    print('usage example: predictFolder.py 86768602162866Data 0.25')
    return
  folderPath = argv[0]
  fraction = float(argv[1])

  filePathList = []

  for filename in os.listdir(folderPath):
    filePathList.append(folderPath + '/' + filename)
  filePathList.sort()

  numberToPredict = int(len(filePathList) * fraction)
  total = 0
  count = 0
  for i in range(len(filePathList) - numberToPredict, len(filePathList)):
    print('progress: %f' %(float(count)/numberToPredict))
    prediction = bms4.main(['predict', filePathList[i]])
    total += prediction
    count += 1
  average = total / (numberToPredict)
  print('prediction: %f' %average)




if __name__ == "__main__":
  main(sys.argv[1:])