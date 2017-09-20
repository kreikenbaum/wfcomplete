'''stores/retrieves results from db

target usage: load from different sources (or upload all to db),
then compute mean and variance/std
'''
import pymongo

import scenario

client = pymongo.MongoClient()
db = client.sacred

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

    def save(self):
        '''saves entry to mongodb'''

if __name__ == "__main__":
    resulting = db.runs.find({"$and": [{"config.scenario": {"$exists": 1}}, {"result.score": {"$exists": 1}}]})
    all_ = [Result.from_mongoentry(x) for x in resulting]
    ## todo: filter to get only one element per scenario
    ### here or mongo?
