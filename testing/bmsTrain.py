import os
import json
import sys
import bms4

'''
Runs bms4.py training on json files in given folder, starting at the beginning
and stopping when the given fraction of files is completed
'''

def main(argv):
  if(len(argv)!=3):
    print('usage: bmsTrain.py <folder filepath> <is authentic> <fraction to train>')
    print('usage example: bmsTrain.py 86768602162866Data 1 0.75')
    return
  folderPath = argv[0]
  isAuthentic = argv[1]
  fraction = float(argv[2])

  filePathList = []

  for filename in os.listdir(folderPath):
    filePathList.append(folderPath + '/' + filename)
  filePathList.sort()

  numberToTrain = len(filePathList) * fraction

  for i in range(0, int(numberToTrain)):
    print('progress: %f' % (i/ numberToTrain))
    bms4.main(['train', filePathList[i], isAuthentic])




if __name__ == "__main__":
  main(sys.argv[1:])