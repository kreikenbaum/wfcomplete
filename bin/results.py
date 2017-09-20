'''stores/retrieves results from db

target usage: load from different sources (or upload all to db),
then compute mean and variance/std
'''
import pymongo

import scenario

client = pymongo.MongoClient()
db = client.sacred

class Result(object):
    def __init__(self, scenario_, accuracy, git):
        self.scenario = scenario_
        self.accuracy = accuracy
        self.git = git

    @staticmethod
    def from_mongoentry(entry):
        return Result(scenario.Scenario(entry['config']['scenario']),
                      entry['result']['score'],
                      entry['experiment']['repositories'][0]['commit'])

if __name__ == "__main__":
    resulting = db.runs.find({"$and": [{"config.scenario": {"$exists": 1}}, {"result.score": {"$exists": 1}}]})
    results = [Result.from_mongoentry(x) for x in resulting]








    
