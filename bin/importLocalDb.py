#! /usr/bin/env python
'''import pre-dumped data from HU

call exportMongo.sh on hosts first, then e.g.
wget -r -np www2.informatik.hu-berlin.de/~kreikenb/data/$(date +%F)
importLocalDb $(date +%F)/*-dump-runs
mongoimport -d sacred -c fs.files $(date +%F)/*-dump-files
mongoimport -d sacred -c fs.chunks $(date +%F)/*-dump-chunks
'''
from dateutil import parser
import fileinput
import json
import logging
import subprocess
import tempfile

import pymongo
import results


def add_to_db(json_line):
    '''add entry in json form to database, add fitting _id'''
    if 'COMPLETED' in json_line:
        try:
            entry = json.loads(json_line)
        except ValueError:
            with open(tempfile.mktemp(), 'w') as f:
                f.write(json_line)
                logging.e("invalid json line: %s, see %s", line[:80], f.name)
        if results._db().runs.count({"stop_time": parser.parse(
                entry['stop_time']['$date'])}):
            logging.warn("seems to already exist: %s", line[:40])
            return
        entry['_id'] = results._next_id()
        imp = subprocess.Popen(
            ["mongoimport", "-c", "runs", "-d", "sacred"],
            stdin=subprocess.PIPE)
        imp.communicate(input=json.dumps(entry))
    else:
        logging.info("skipped line starting with %s", line[:40])


if __name__ == "__main__":
    try:
        pymongo.MongoClient(serverSelectionTimeoutMS=10).server_info()
    except pymongo.errors.ServerSelectionTimeoutError:
        print 'start local mongodb!'
        raise

    for line in fileinput.input():
        add_to_db(line)
