import glob
import re
import csv

path = "*.txt"
for fname in glob.glob(path):
    #hand = open(fname)
    hand = open(fname, 'r')
    refilename1='((?:[a-z][a-z]+))'	# Word 1
    refilename2='.*?'	# Non-greedy match on filler
    refilename3='(?:[a-z][a-z]+)'	# Uninteresting: word
    refilename4='.*?'	# Non-greedy match on filler
    refilename5='((?:[a-z][a-z]+))'	# Word 2

    rg = re.compile(refilename1+refilename2+refilename3+refilename4+refilename5,re.IGNORECASE|re.DOTALL)
    m = rg.search(fname)
    if m:
        filenameword1=m.group(1)
        filenameword2=m.group(2)
        # print "("+filenameword1+")"+"("+filenameword2+")"+"\n"

        for line in hand:
            re1='((?:[a-z][a-z0-9_]*))'	# Variable Name 1
            re2='((?:\\/[\\w\\.\\-]+)+)'	# Unix Path 1
            re3='.*?'	# Non-greedy match on filler
            re4='((?:(?:[1]{1}\\d{1}\\d{1}\\d{1})|(?:[2]{1}\\d{3}))[-:\\/.](?:[0]?[1-9]|[1][012])[-:\\/.](?:(?:[0-2]?\\d{1})|(?:[3][01]{1})))(?![\\d])'	# YYYYMMDD 1
            re5='.*?'	# Non-greedy match on filler
            re6='((?:(?:[0-1][0-9])|(?:[2][0-3])|(?:[0-9]))((?:_)|(?::))(?:[0-5][0-9])(?::[0-5][0-9])?(?:\\s?(?:am|AM|pm|PM))?)'	# HourMinuteSec 1
            re7='.*?'	# Non-greedy match on filler
            re8='([+-]?\\d*\\.\\d+)(?![-+0-9\\.])'	# Float 1

            rg2 = re.compile(re1+re2+re3+re4+re5+re6+re7+re8,re.IGNORECASE|re.DOTALL)
            m2 = rg2.search(line)
            if m2:
                var1=m2.group(1)
                unixpath1=m2.group(2)
                yyyymmdd1=m2.group(3)
                time1=m2.group(4)
                float1=m2.group(6)
                date = yyyymmdd1.split('-')
                time = time1[:2]+':'+time1[3:]
                with open(filenameword1+"_"+filenameword2+"_total.csv", 'a') as f:
                    print "("+filenameword1+")"+"("+filenameword2+")"+"\n"
                    writer = csv.writer(f)
                    writer.writerow([date[1]+'/'+date[2]+'/'+date[0], time, float1])
                f.close()
    hand.close()
