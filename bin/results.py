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
    def __init__(self, scenario_, accuracy, git, time, status):
        self.scenario = scenario_
        self.cumul = accuracy
        self.git = git
        self.time = time
        self.status = status

    @staticmethod
    def from_mongoentry(entry):
        try:
            git = entry['experiment']['repositories'][0]['commit']
        except KeyError:
            git = None
        return Result(scenario.Scenario(entry['config']['scenario']),
                      entry['result']['score'],
                      git,
                      entry['stop_time'],
                      entry['status'])

    def __add__(self, other):
        return Result(None, self.accuracy + other.accuracy, None, None)

    def __repr__(self):
        return '<Result({}, {}, {}, {}, {})>'.format(
            self.scenario, self.cumul, self.git, self.time, self.status)

    def save(self):
        '''saves entry to mongodb if new'''
        obj = {"config": {"scenario": self.scenario},
               "result": {"score": self.cumul, "type": self.status},
               "stop_time": self.time, "status": "EXTERNAL"}
        if db.runs.count(obj) == 0:
            db.runs.insert_one(obj)
        else:
            logging.warn("%s@%s already in db", self.scenario, self.time)

def import_to_mongo(csvfile):
    imported = []
    for el in csv.DictReader(csvfile):
        try:
            date = datetime.datetime.strptime(el['notes'], '%Y-%m-%d')
        except ValueError:
            date = None
        if el['cumul']:
            imported.append(Result(el['defense'], el['cumul'], None,
                                   date, "cumul"))
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
