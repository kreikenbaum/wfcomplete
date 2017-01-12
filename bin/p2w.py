#!/usr/bin/env python
'''create batch/ -dir from large panchenko directory. This is intended
for large directories. Usage f.ex. 

$ python p2w.py foreground-data/output-tcp
'''
import os
import sys

import counter

try:
    os.mkdir("batch")
except OSError:
    os.remove("batch_list")
batch_list = open("batch_list", "w")
url_id = 0
for (url, data) in counter.panchenko_generator(sys.argv[1]):
    if not data:
        continue
    batch_list.write("{}: {}\n".format(url_id, url))
    for (counter_id, datum) in enumerate(data):
        datum.to_wang(os.path.join(
            "batch", "{}-{}".format(url_id, counter_id)))
    url_id += 1
batch_list.close()
