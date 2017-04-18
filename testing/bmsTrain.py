import os
import json
import sys
import bms4


'''
Runs bms4.py training on json files in given folder, starting at the beginning
and stopping when the given fraction of files is completed
'''

def deleteEvents(folderPath):
  numDeleted = 0
  for filename in os.listdir(folderPath):
    filepath = folderPath + '/' + filename
    if(filename.startswith('events.out.tfevents')):
      os.remove(filepath)
      numDeleted += 1
  print('deleted {0} files'.format(numDeleted))

def main(argv):
  if(len(argv)!=4):
    print('usage: bmsTrain.py <folder filepath> <is authentic> <start fraction> <end fraction')
    print('usage example: bmsTrain.py 86768602162866Data 1 0.75')
    return
  folderPath = argv[0]
  isAuthentic = argv[1]
  startFraction = float(argv[2])
  endFraction = float(argv[3])

  filePathList = []

  for filename in os.listdir(folderPath):
    filePathList.append(folderPath + '/' + filename)
  filePathList.sort()

  startIndex = len(filePathList) * startFraction
  endIndex = len(filePathList) * endFraction

  for i in range(int(startIndex), int(endIndex)):
    print('progress: %f' % ((i - startIndex)/ (endIndex- startIndex)))
    bms4.main(['train', filePathList[i], isAuthentic])
    if(i%10 == 0):
      deleteEvents('bms4_model')
  deleteEvents('bms4_model')






if __name__ == "__main__":
  main(sys.argv[1:])