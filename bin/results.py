'''stores/retrieves results from db

target usage: load from different sources (or upload all to db),
then compute mean and variance/std
'''
import csv
import datetime
import logging

import pymongo

import scenario

class Result(object):
    def __init__(self, scenario_, accuracy, git, time, status, size):
        self.scenario = scenario_
        self.cumul = accuracy
        self.git = git
        self.date = time
        if not self.date and hasattr(self.scenario, "date"):
            self.date = self.scenario.date
        self.status = status
        self.size = size

    @staticmethod
    def from_mongoentry(entry):
        try:
            git = entry['experiment']['repositories'][0]['commit']
        except KeyError:
            git = None
        try:
            size = len(entry['result']['sites'])
        except KeyError:
            size = entry['config']['size']
        return Result(scenario.Scenario(entry['config']['scenario']),
                      entry['result']['score'],
                      git,
                      entry['stop_time'],
                      entry['status'],
                      size)

    def __add__(self, other):
        return Result(None, self.accuracy + other.accuracy, None, None)

    def __repr__(self):
        return '<Result({!r}, {}, {}, {}, {})>'.format(
            self.scenario, self.cumul, self.git, self.date, self.status)

    def save(self):
        '''saves entry to mongodb if new'''
        obj = {"config": {"scenario": self.scenario, "size": self.size},
               "result": {"score": self.cumul, "type": self.status},
               "stop_time": self.date, "status": "EXTERNAL"}
        if db.runs.count(obj) == 0:
            db.runs.insert_one(obj)
        else:
            logging.debug("%s@%s already in db", self.scenario, self.date)

def import_to_mongo(csvfile, size):
    imported = []
    for el in csv.DictReader(csvfile):
        try:
            date = datetime.datetime.strptime(el['notes'], '%Y-%m-%d')
        except ValueError:
            date = None
        if el['cumul']:
            imported.append(Result(el['defense'], el['cumul'], None,
                                   date, "cumul", size))
        else:
            logging.warn("skipped %s", el)
    for el in imported:
        #print el
        el.save()

if __name__ == "__main__":
    client = pymongo.MongoClient()
    db = client.sacred
    all_ = [Result.from_mongoentry(x) for x in
            db.runs.find({"$and": [{"config.scenario": {"$exists": 1}},
                                   {"result.score": {"$exists": 1}}]})]
    ## todo: filter to get only one element per scenario
    ### here or mongo?
    # import_to_mongo(open('/home/uni/da/git/data/results/10sites.csv'))
