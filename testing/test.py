def stringToBits(str, numBits):
  result = []
  h = 13
  length = len(str)
  for i in range(length):
    h = 31 * h + ord(str[i])
  #h has become an int representation of the string
  #take repeated modulo two to get bits
  for _ in range(numBits):
    result.append(h%2)
    h/=2
  return result

print stringToBits('abc', 8)
print stringToBits('abcd', 8)