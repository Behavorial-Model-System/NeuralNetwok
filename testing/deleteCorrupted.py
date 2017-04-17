import os
import json
import sys

'''
Goes through a folder and deletes json files that are invalid
'''



def main(argv):
  if(len(argv)!=1):
    print('usage: deleteCorruptedScript.py <folder filepath>')
    return
  folderPath = argv[0]

  numDeleted = 0
  for filename in os.listdir(folderPath):
    filepath = folderPath + '/' + filename
    if(not filename.endswith('.json')):
      continue
    with open(filepath) as data_file:
      try:
        json.load(data_file)
      except:
        print(filename + ' could not be loaded and will be deleted')
        os.remove(filepath)
        numDeleted += 1

  print('number of deleted json files: %d' %numDeleted)


if __name__ == "__main__":
  main(sys.argv[1:])