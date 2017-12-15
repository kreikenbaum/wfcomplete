#! /usr/bin/env python
'''imports experiment data from duckstein into local db'''
import json
import os
import results
import subprocess

import pymongo

try:
    pymongo.MongoClient(serverSelectionTimeoutMS=10).server_info()
except pymongo.errors.ServerSelectionTimeoutError:
    print 'start local mongodb!'
    exit(1)

runs = subprocess.check_output(
    ["ssh", "mkreik@duckstein", "mongoexport -d sacred -c runs"])

for line in runs.split('\n'):
    if 'COMPLETED' in line:
        entry = json.loads(line)
        entry['_id'] = results._next_id()
        imp = subprocess.Popen(["mongoimport", "-c", "runs", "-d", "sacred"],
                               stdin=subprocess.PIPE)
        imp.communicate(input=json.dumps(entry))

os.system("ssh mkreik@duckstein 'mongoexport -d sacred -c runs > /home/mkreik/data/mongodump$(date +%F_%T)'")
subprocess.call(
    ["ssh", "mkreik@duckstein",
     'mongo sacred --eval "printjson(db.dropDatabase())"'])


