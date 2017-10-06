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
                 size_overhead=None, time_overhead=None):
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

    @staticmethod
    def from_mongoentry(entry):
        git = _nestedvalue_or_none(entry, 'experiment', 'repositories', 0, 'commit')
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
        size_overhead = _nestedvalue_or_none(entry, 'result', 'size_increase')
        time_overhead = _nestedvalue_or_none(entry, 'result', 'time_increase')
        return Result(scenario.Scenario(entry['config']['scenario']),
                      entry['result']['score'],
                      git,
                      entry['stop_time'],
                      type_,
                      size,
                      size_overhead,
                      time_overhead)

    def __add__(self, other):
        return Result(None, self.accuracy + other.accuracy, None, None)

    def __repr__(self):
        return '<Result({!r}, {}, {}, {}, {}, size={}, size_overhead={}, time_overhead={})>'.format(
            self.scenario, self.cumul, self.git, self.date,
            self.type_, self.size, self.size_overhead, self.time_overhead)

    def save(self):
        '''saves entry to mongodb if new'''
        obj = {
            "_id": _next_id(),
            "config": {"scenario": self.scenario, "size": self.size},
            "result": {"score": self.cumul, "type": self.type_},
            "stop_time": self.date, "status": "EXTERNAL"}
        if db.runs.count(obj) == 0:
            db.runs.insert_one(obj)
        else:
            logging.debug("%s@%s already in db", self.scenario, self.date)

        
def _nestedvalue_or_none(entry, *steps):
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
        #print el
        el.save()

def get_all(match={"$and": [{"config.scenario": {"$exists": 1}},
                            {"result.score": {"$exists": 1}}]}):
    '''@return all runs that match match'''
    db = pymongo.MongoClient().sacred
    return [Result.from_mongoentry(x) for x in
            db.runs.find(match)]

def to_table(results):
    '''@return table of results as a string'''
    tbl = prettytable.PrettyTable()
    tbl.field_names = ["scenario", "accuracy [in %]", "size increase [in %]",
                       "time increase [in %]"]
    for result in results:
        tbl.add_row(name, score, si, ti)
        
def sized(size):
    return [x for x in get_all() if x.size == size]
#    return get_all({"$or": [{"config.size": 10},
#                            {"result.sites": {"$size": 10}}]})

if __name__ == "__main__":
    db = pymongo.MongoClient().sacred
    all_ = [Result.from_mongoentry(x) for x in
            db.runs.find({"$and": [{"config.scenario": {"$exists": 1}},
                                   {"result.score": {"$exists": 1}}]})]
    ## todo: filter to get only one element per scenario
    ### here or mongo?
    # import_to_mongo(open('/home/uni/da/git/data/results/10sites.csv'), 10)


# todo = filter(lambda x: (x.size == 30 and "disabled" not in x.scenario.path and x.size_overhead is None), get_all())
# for t in todo: os.system('''~/da/git/bin/exp.py -e with 'scenario = "{}"' '''.format(t.scenario.path))
