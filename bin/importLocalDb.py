#! /usr/bin/env python
'''import pre-dumped data from HU'''
# (need to check that file)
# 0.1 on hosts: mongoexport -d sacred -c runs > data/$(hostname)-dump
# 0.2 scp star.informatik.hu-berlin.de:data/*dump to e.g. /tmp
# 1 import local files
# 0.3 mv data/dump to data/old$(date +%F_%T)
#  - mkdir data/old...
#  - mv data/*dump data/old...
# 0.4 clear databases: mongo sacred --eval "printjson(db.dropDatabase())"
#  - if failure on gruenau5 (running exp): code to check if experiment running
import fileinput
import json
import logging
import subprocess
import tempfile

import pymongo
import results


def add_to_db(json_line):
    if 'COMPLETED' in json_line:
        try:
            entry = json.loads(line)
        except ValueError:
            with open(tempfile.mktemp(), 'w') as f:
                f.write(json_line)
                logging.e("invalid json line: %s, see %s", line[:80], f.name)
        entry['_id'] = results._next_id()
        imp = subprocess.Popen(
            ["mongoimport", "-c", "runs", "-d", "sacred"],
            stdin=subprocess.PIPE)
        imp.communicate(input=json.dumps(entry))
    else:
        logging.info("skipped line starting with %s", line[:80])


if __name__ == "__main__":
    try:
        pymongo.MongoClient(serverSelectionTimeoutMS=10).server_info()
    except pymongo.errors.ServerSelectionTimeoutError:
        print 'start local mongodb!'
        raise

    for line in fileinput.input():
        add_to_db(line)
