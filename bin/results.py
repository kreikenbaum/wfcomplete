'''stores/retrieves results from db

target usage: load from different sources (or upload all to db),
then compute mean and variance/std
'''
import csv
import datetime
import logging

import pymongo
import prettytable

import scenario

class Result(object):
    def __init__(self, scenario_, accuracy, git, time, type_, size,
                 size_overhead=None, time_overhead=None, _id=None):
        self.scenario = scenario_
        self.cumul = accuracy
        self.git = git
        self.date = time
        if not self.date and hasattr(self.scenario, "date"):
            self.date = self.scenario.date
        self.type_ = type_
        self.size = size
        self.size_overhead = size_overhead
        self.time_overhead = time_overhead
        self._id = _id

    @staticmethod
    def from_mongoentry(entry):
        git = _value_or_none(entry, 'experiment', 'repositories', 0, 'commit')
        try:
            size = len(entry['result']['sites'])
        except KeyError:
            size = entry['config']['size']
        try:
            type_ = entry['result']['type']
        except KeyError:
            if entry['status'] == 'COMPLETED':
                type_ = "cumul"
            else:
                raise
        size_overhead = _value_or_none(entry, 'result', 'size_increase')
        time_overhead = _value_or_none(entry, 'result', 'time_increase')
        return Result(scenario.Scenario(entry['config']['scenario']),
                      entry['result']['score'],
                      git,
                      entry['stop_time'],
                      type_,
                      size,
                      size_overhead,
                      time_overhead,
                      entry['_id'])

    def __add__(self, other):
        return Result(None, self.accuracy + other.accuracy, None, None)

    def __repr__(self):
        return '<Result({!r}, {}, {}, {}, {}, size={}, size_overhead={}, time_overhead={})>'.format(
            self.scenario, self.cumul, self.git, self.date,
            self.type_, self.size, self.size_overhead, self.time_overhead)

    def save(self):
        '''saves entry to mongodb if new'''
        db = pymongo.MongoClient().sacred
        obj = {
            "_id": _next_id(),
            "config": {"scenario": self.scenario, "size": self.size},
            "result": {"score": self.cumul, "type": self.type_},
            "stop_time": self.date, "status": "EXTERNAL"}
        if db.runs.count(obj) == 0:
            db.runs.insert_one(obj)
        else:
            logging.debug("%s@%s already in db", self.scenario, self.date)


    # add overhead-helper for those experiments that lacked it
    def _add_oh(self):
        self.update(
            {"result.size_increase": self.scenario.size_increase(),
             "result.time_increase": self.scenario.time_increase()})
    def update(self, addthis):
        '''updates entry in db to also include addthis'''
        db = pymongo.MongoClient().sacred
        db.runs.find_one_and_update({"_id": self._id}, {"$set": addthis})
        
def _value_or_none(entry, *steps):
    '''descends steps into entry, returns value or None if it does not exist'''
    try:
        for step in steps:
            entry = entry[step]
        return entry
    except KeyError:
        return None


def _next_id():
    '''@return next id for entry to db'''
    return (db.runs
            .find({}, {"_id": 1})
            .sort("_id", pymongo.DESCENDING)
            .limit(1)
            .next())["_id"] + 1


def import_to_mongo(csvfile, size, measure="cumul"):
    imported = []
    for el in csv.DictReader(csvfile):
        try:
            date = datetime.datetime.strptime(el['notes'], '%Y-%m-%d')
        except ValueError:
            date = None
        if el[measure]:
            imported.append(Result(el['defense'], float(el[measure]), None,
                                   date, measure, size))
        else:
            logging.warn("skipped %s", el)
    for el in imported:
        el.save()

def list_all(match=None, restrict=True):
    '''@return all runs (normally with scenario and score) that match match'''
    matches = [{"config.scenario": {"$exists": 1}},
               {"result.score": {"$exists": 1}}] if restrict else []
    if match: matches.append(match)
    db = pymongo.MongoClient().sacred
    return [Result.from_mongoentry(x) for x in
            db.runs.find({"$and": matches})]


def to_table(results):
    '''@return table of results as a string'''
    tbl = prettytable.PrettyTable()
    tbl.field_names = ["scenario", "accuracy [in %]", "size increase [in %]",
                       "time increase [in %]"]
    for result in results:
        tbl.add_row(name, score, si, ti)
    TODO
        
def sized(size):
    return [x for x in list_all() if x.size == size]
#    return list_all({"$or": [{"config.size": 10},
#                            {"result.sites": {"$size": 10}}]})

def _duplicates(params=["config.scenario", "result.score"]):
    '''@return all instances of experiments, projected to only params
    >>> {x['_id']: x['result_score'] for x in _duplicates()}.iteritems().next()
    (u'0.22/5aI--2016-07-25', [0.691191114775])
    '''
    db = pymongo.MongoClient().sacred
    project = {key: 1 for key in params}
    groups = {"_id": "$config.scenario", "count": {"$sum": 1}}
    local = params[:]
    local.remove('config.scenario')
    for each in local:
        groups[each.replace(".", "_")] = {"$push": "${}".format(each)}
    return db.runs.aggregate([
        {"$match": {"$and": [
            {"config.scenario": {"$exists": 1}},
            {"result.score": {"$exists": 1}}]}},
        {"$project": project},
        {"$group": groups}])


if __name__ == "__main__":
    db = pymongo.MongoClient().sacred
    all_ = [Result.from_mongoentry(x) for x in
            db.runs.find({"$and": [{"config.scenario": {"$exists": 1}},
                                   {"result.score": {"$exists": 1}}]})]
    ## todo: filter to get only one element per scenario
    ### here or mongo?
    # import_to_mongo(open('/home/uni/da/git/data/results/10sites.csv'), 10)


    # a = list(_duplicates(["config.scenario", "result.score", "result.size_increase", "result.time_increase", "status"]))
    # todos = [x['_id'] for x in a if len(x['result_size_increase']) == 0]
    # for t in todo: os.system('''~/da/git/bin/exp.py -e with 'scenario = "{}"' '''.format(t.scenario.path))
    # for t in todos: print(t); list_all({"config.scenario": t})[-1]._add_oh()
    # # some non-existing ones were pop()ped from todos
    #db.runs.update({_id: 23}, {$set: {"stabus": "FABLED"}})
    
#  result with closest size overhead
# b = [x for x in get_all() if x.scenario.name == '0.22' and x.size_overhead]
# min(b, key=lambda x: abs(size - x.size_overhead))
# c = [x for x in get_all() if x.scenario.name == '0.22' and x.time_overhead]
# min(c, key=lambda x: abs(42 - x.time_overhead))
# d = [x for x in get_all() if x.scenario.name == '0.22']
# min(d, key=lambda x: abs(0.63 - x.cumul))
# min(d, key=lambda x: abs(0.26 - x.cumul))
