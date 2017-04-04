
#converts string to int
def stringToBits(str, numBits):
    h = 13
    length = len(str)
    for i in range(length):
        h = 31*h + ord(str[i])
    return h