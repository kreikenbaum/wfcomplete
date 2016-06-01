import fileinput
import json

if __name__ == "__main__":
    data = {}
    for line in fileinput.input():
        (url, val) = line.split(': ')
        if '?' in url:
            url = url.split('?')[0]
        # always choose lower value for security
        # td?: warn on duplicate?
        if url in data.keys() and data[url] < val:
            pass
        else:
            data[url] =  int(val)
    print json.dumps(list(data.iteritems()))

# get quantiles
# import numpy as np
# a = data.values()
# a.sort()
# for i in range(1, 6):
#     print '{}: {}'.format(i/6.0, np.percentile(b, 100.0*i/6.0))


