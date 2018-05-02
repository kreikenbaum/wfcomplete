#! /usr/bin/env python
'''imports experiment data from host (default:duckstein) into local db

this needs mongoexport and mongo to be in path'''
import json
import os
import results
import subprocess
import sys

import pymongo

DEFAULT_LOGIN = "mkreik@duckstein"


# ## duplicate code with importLocalDb
def load(ssh_login):
    try:
        pymongo.MongoClient(serverSelectionTimeoutMS=10).server_info()
    except pymongo.errors.ServerSelectionTimeoutError:
        print 'start local mongodb!'
        exit(1)

    runs = subprocess.check_output(
        ["ssh", DEFAULT_LOGIN, "mongoexport -d sacred -c runs"])

    for line in runs.split('\n'):
        #        if 'COMPLETED' in line:
        if not line:
            continue
        else:
            entry = json.loads(line)
            entry['_id'] = results._next_id()
            imp = subprocess.Popen(
                ["mongoimport", "-c", "runs", "-d", "sacred"],
                stdin=subprocess.PIPE)
            imp.communicate(input=json.dumps(entry))

    os.system("ssh {} 'mongoexport -d sacred -c runs > /home/mkreik/data/mongodump$(date +%F_%T)'".format(DEFAULT_LOGIN))
    subprocess.call(
        ["ssh", DEFAULT_LOGIN,
         'mongo sacred --eval "printjson(db.dropDatabase())"'])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        load(sys.argv[1])
    else:
        load(DEFAULT_LOGIN)
