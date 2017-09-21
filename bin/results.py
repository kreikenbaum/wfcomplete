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
    def __init__(self, scenario_, accuracy, git, time):
        self.scenario = scenario_
        self.cumul = accuracy
        self.git = git
        self.time = time

    @staticmethod
    def from_mongoentry(entry):
        return Result(scenario.Scenario(entry['config']['scenario']),
                      entry['result']['score'],
                      entry['experiment']['repositories'][0]['commit'],
                      entry['stop_time'])

    def __add__(self, other):
        return Result(None, self.accuracy + other.accuracy, None, None)

    def __repr__(self):
        return '<Result({}, {}, {}, {})>'.format(
            self.scenario, self.cumul, self.git, self.time)

    def save(self):
        '''saves entry to mongodb'''
        db.runs.insert_one({"config": {"scenario": self.scenario},
                            "result": {"score": self.cumul},
                            "stop_time": self.time,
                            "status": "EXTERNAL"})

def import_to_mongo(csvfile):
    imported = []
    for el in csv.DictReader(csvfile):
        try:
            date = datetime.datetime.strptime(el['notes'], '%Y-%m-%d').date()
        except ValueError:
            date = None
        if el['cumul']:
            imported.append(Result(el['defense'], el['cumul'], None, date))
        else:
            logging.warn("skipped {}".format(str(el)))
    for el in imported:
        print el
        #el.save()

if __name__ == "__main__":
    client = pymongo.MongoClient()
    db = client.sacred
    all_ = [Result.from_mongoentry(x) for x in
            db.runs.find({"$and": [{"config.scenario": {"$exists": 1}},
                                   {"result.score": {"$exists": 1}}]})]
    ## todo: filter to get only one element per scenario
    ### here or mongo?
    # import_to_mongo(open('/home/uni/da/git/data/results/10sites.csv'))
