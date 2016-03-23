import fileinput
import json

data = []
for line in fileinput.input():
    (val, url) = line.split()
    data.append((url, int(val)))
print json.dumps(data)
