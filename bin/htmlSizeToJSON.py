#! /usr/bin/env python
'''@returns the (max) lengths of each html file'''
import fileinput
import json

# def sanityCheck(data):
#     '''tests for duplicates urls'''
#     urls = [d[0] for d in data]
#     if len(urls) != len(set(urls)):
#         raise Exception("duplicate url")
    
if __name__ == "__main__":
    data = {}
    for line in fileinput.input():
        (val, url) = line.split()
        # always choose lower value for security
        if url in data.keys() and data[url] < val:
            pass
        else:
            data[url] =  int(val)
    print json.dumps(list(data.iteritems()))

