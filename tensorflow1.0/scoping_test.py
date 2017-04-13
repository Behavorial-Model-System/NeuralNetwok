
size = 1
class heightObject:
  height = 1

def func():
  print('size %d' %size)
  print('height %d' %heightObject.height)

def main():
  heightObject.height = 2
  size = 2
  func()

if __name__ == "__main__":
  main()
