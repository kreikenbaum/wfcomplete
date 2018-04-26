#! /usr/bin/env python
'''stores/retrieves results from db

target usage: load from different sources (or upload all to db),
then compute mean and variance/std
'''
from __future__ import print_function
import csv
import collections
import datetime
import doctest
import logging

import numpy as np
import pymongo
from sklearn import metrics, model_selection

import scenario
from capture import utils
import config


def _db():
    return pymongo.MongoClient(serverSelectionTimeoutMS=10).sacred


class LastIter(object):
    def __init__(self, iter_):
        self.iter = iter(iter_)

    def __iter__(self):
        return self.iter

    def next(self):
        return self.iter.next()

    def last(self):
        tmp = None
        for _ in self:
            tmp = _
        return tmp


class Result(object):
    def __init__(self, scenario_, accuracy, git, time, type_, size,
                 open_world=False, size_overhead=None,
                 time_overhead=None, _id=None, gamma=None, C=None, src=None,
                 ytrue=None, ypred=None, ydomains=None):
        self._id = _id
        self.C = C
        self.date = time
        self.gamma = gamma
        self.git = git
        self.open_world = open_world
        self.scenario = scenario_
        self.score = accuracy
        self.size = size
        self.size_overhead = size_overhead
        self.time_overhead = time_overhead
        self.type_ = type_
        if not self.date and hasattr(self.scenario, "date"):
            self.date = self.scenario.date
        self.src = src
        self._ytrue = ytrue
        self._ypred = ypred
        self._ydomains = ydomains

    @property
    def background_size(self, compute=False):
        '''@return background size, real if available, else config parameter'''
        assert self.open_world
        if self._ytrue or compute:
            return collections.Counter(self.y_true)[-1]
        else:
            return self.open_world['background_size']

    @property
    def duration(self):
        '''@return experiment duration, or -1'''
        try:
            return self.src['stop_time'] - self.src['start_time']
        except KeyError:
            return None

    @property
    def host(self):
        '''@return name of host doing experiment'''
        try:
            return self.src['host']['hostname']
        except KeyError:
            return "unknown"

    @property
    def y_domains(self):
        '''@return domains of traces, with optional open world added'''
        if self._ydomains is None:
            logging.warn("%s had no saved domain array 'yd'", self)
            _, _, self._ydomains = self.scenario.get_features_cumul(
                self.open_world['current_sites'])
        return self._ydomains

    @property
    def y_prediction(self):
        '''@return predicted values (either pre-existing or computed)'''
        if not self._ypred:
            logging.warn("%s had no saved prediction", self)
            X, y, _ = self.scenario.get_features_cumul(
                self.open_world['current_sites'])
            self._ypred = model_selection.cross_val_predict(
                self.get_classifier(), X, y, cv=config.FOLDS,
                n_jobs=config.JOBS_NUM, verbose=config.VERBOSE)
        return np.array(self._ypred)

    @property
    def y_true(self):
        '''@return true classes of traces, with optional open world added'''
        if self._ytrue is None:
            logging.warn("%s had no saved class array 'y'", self)
            _, self._ytrue, _ = self.scenario.get_features_cumul(
                self.open_world['current_sites'])
        return self._ytrue

    @staticmethod
    def from_mongoentry(entry):
        git = _value_or_none(entry, 'experiment', 'repositories', 0, 'commit')
        c = (_value_or_none(entry, 'result', 'C') or
             _value_or_none(entry, 'result', 'clf', 'py/state', 'estimator',
                            'py/state', 'C'))
        gamma = (_value_or_none(entry, 'result', 'gamma') or
                 _value_or_none(entry, 'result', 'clf', 'py/state',
                                'estimator', 'py/state', 'gamma'))
        try:
            size = len(entry['result']['sites'])
        except (KeyError, TypeError):
            size = _value_or_none(entry, 'config', 'size')
        try:
            type_ = _value_or_none(entry, 'result', 'type')
        except KeyError:
            if entry['status'] == 'COMPLETED':
                type_ = "cumul"
            else:
                raise
        try:
            open_world = entry['experiment']['name'] == 'wf_open_world'
            if open_world:  # non-empty dict is True
                open_world = {
                    'fpr': _value_or_none(entry, 'result', 'fpr'),
                    'tpr': _value_or_none(entry, 'result', 'tpr'),
                    'auroc': _value_or_none(entry, 'result', 'auroc'),
                    'auc_bound': _value_or_none(entry, 'config', 'auc_bound'),
                    'background_size': _value_or_none(
                        entry, 'config', 'background_size'),
                    'binary': _value_or_none(entry, 'config', 'binarize'),
                    'exclude_sites': _value_or_(
                        entry, [], 'config', 'exclude_sites'),
                    'current_sites': _value_or_none(
                        entry, 'config', 'current_sites')
                }
        except KeyError:
            open_world = False
        size_overhead = _value_or_none(entry, 'result', 'size_increase')
        time_overhead = _value_or_none(entry, 'result', 'time_increase')
        try:
            orl = _value_or_none(entry, 'config', 'or_level')
            config.OR_LEVEL = config.OR_LEVEL if orl is None else orl
            rems = _value_or_none(entry, 'config', 'remove_small')
            config.REMOVE_SMALL = config.REMOVE_SMALL if rems is None else rems
            scenario_obj = scenario.Scenario(
                entry['config']['scenario'], open_world=open_world)
            reload(config)
        except ValueError:
            if entry['status'] not in ["COMPLETED", "EXTERNAL"]:
                scenario_obj = "placeholder for scenario {}".format(
                    entry['config']['scenario'])
            else:
                raise
        yt = (_value_or_none(entry, 'result', 'y_true', 'values')
              or _value_or_none(entry, 'result', 'y_true'))
        return Result(
            scenario_obj,
            _value_or_none(entry, 'result', 'score'),
            git,
            _value_or_none(entry, 'stop_time'),
            type_,
            size,
            open_world,
            size_overhead=size_overhead, time_overhead=time_overhead,
            _id=entry['_id'],
            C=c, gamma=gamma,
            src=entry,
            ytrue=yt,
            ypred=_value_or_none(entry, 'result', 'y_prediction'),
            ydomains=_value_or_none(entry, 'result', 'y_domains'))

    def __repr__(self):
        out = ("<Result({!r}, score={}, {}, {}, {}, size={}, "
               "size_overhead={}, time_overhead={}".format(
                   self.scenario, self.score, self.git, self.date, self.type_,
                   self.size, self.size_overhead, self.time_overhead))
        if self.open_world:
            out += ', open_world={}'.format(self.open_world)
        return out + ')>'

    def __str__(self):
        return "Result {} for {}".format(self._id, self.scenario)

    def get_classifier(self, probability=True):
        '''@return classifier that achieved this result'''
        return utils.clf_default(
            C=self.C, gamma=self.gamma,
            class_weight=None if self.open_world else "balanced",
            probability=probability)

    def get_confusion_matrix(self):
        '''@return confusion matrix, from pre-existing or computed values'''
        return metrics.confusion_matrix(self.y_true, self.y_prediction)

    def save(self, db=_db()):
        '''saves entry to mongodb if new'''
        obj = {
            "_id": _next_id(),
            "config": {"scenario": self.scenario, "size": self.size},
            "result": {"score": self.score, "type": self.type_},
            "stop_time": self.date, "status": "EXTERNAL"}
        if db.runs.count(obj) == 0:
            db.runs.insert_one(obj)
        else:
            logging.debug("%s@%s already in db", self.scenario, self.date)

    # df = pd.DataFrame(r.to_dict() for r in results.list_all())
    def to_dict(self):
        '''@return dict version of this, for use in e.g. pandas'''
        out = {}
        # add _type if other types than "accuracy"
        for key in ['C', 'size_overhead', 'time_overhead', 'score',
                    'date', '_id', 'gamma', 'size']:
            out[key] = self.__dict__[key]
        if self.open_world:
            for key in self.open_world:
                out[key] = self.open_world[key]
        for prop in ['background_size', 'duration']:  # bg:overwrite open_world
            try:
                out[prop] = getattr(self, prop)
            except AssertionError:  # fail for closed-world
                pass
        try:
            for key in ['name', 'date', 'path']:  # (+num_sites etc)
                out["scenario." + key] = self.scenario.__dict__[key]
        except KeyError:
            pass  # ok to not include (e.g. WANG14 external result)
        return out

    # def _add_oh(self):
    #     '''add overhead-helper for those experiments that lacked it'''
    #     self.update(
    #         {"result.size_increase": self.scenario.size_increase(),
    #          "result.time_increase": self.scenario.time_increase()})
    def update(self, addthis, db=_db()):
        '''updates entry in db to also include addthis'''
        db.runs.find_one_and_update({"_id": self._id}, {"$set": addthis})


def _duplicates(params=["config.scenario", "result.score"], db=_db()):
    '''@return all instances of experiments, projected to only params
    example: {x['_id']: x['result_score'] for x in _duplicates()}
    '''
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


def _next_id(db=_db()):
    '''@return next id for entry to db'''
    return (db.runs
            .find({}, {"_id": 1})
            .sort("_id", pymongo.DESCENDING)
            .limit(1)
            .next())["_id"] + 1


def _value_or_(entry, ifnot, *steps):
    try:
        for step in steps:
            entry = entry[step]
        return entry
    except (KeyError, IndexError, TypeError):
        return ifnot


def _value_or_none(entry, *steps):
    '''descends steps into entry, returns value or None if it does not exist'''
    return _value_or_(entry, None, *steps)


def for_id(_id):
    '''@return result with id _id'''
    return (x for x in list_all() if x._id == _id).next()



def for_scenario(scenario_obj):
    return (x for x in list_all() if x.scenario == scenario_obj)


def for_scenario_closed(scenario_obj):
    return (x for x in for_scenario(scenario_obj) if not x.open_world)


def for_scenario_open(scenario_obj):
    return (x for x in for_scenario(scenario_obj) if x.open_world)


def for_scenario_smartly(scenario_obj):
    if scenario_obj.open_world:
        out = for_scenario_open(scenario_obj)
        if scenario_obj.binary:
            return (x for x in out if x.open_world['binary'])
        else:
            return (x for x in out if not x.open_world['binary'])
    return for_scenario_closed(scenario_obj)



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


def list_all(match=None, restrict=True, db=_db()):
    '''@return all runs (normally with scenario and score) that match match'''
    matches = [{"config.scenario": {"$exists": 1}}]
    if restrict:
        matches.append({"result.score": {"$exists": 1}})
    if match:
        matches.append(match)
    return LastIter(
        Result.from_mongoentry(x) for x in db.runs.find({"$and": matches}))


def sized(size):
    '''@return results with size size'''
    return [x for x in list_all() if x.size == size]
#    return list_all({"$or": [{"config.size": 10},
#                            {"result.sites": {"$size": 10}}]})


def to_table(results, fields_plus_names=None):
    '''@return table of results as a string'''
    if fields_plus_names:
        (fields, names) = fields_plus_names
    else:
        def fields(r):
            return [r.scenario.name, r.date.date(),
                    r.score, r.size_overhead, r.time_overhead]
        names = ["scenario", "date",
                 "accuracy [%]", "size increase [%]", "time increase [%]"]
    import prettytable
    tbl = prettytable.PrettyTable()
    tbl.field_names = names
    # names if names else fields
    for result in results:
        tbl.add_row(fields(result))
    return tbl


doctest.testmod(optionflags=doctest.ELLIPSIS)

if __name__ == "__main__":
    print(_next_id())


# ### import csv to mongodb
# import_to_mongo(open('/home/uni/da/git/data/results/10sites.csv'), 10)

# ### add overheads to each scenario once
# a = list(_duplicates(["config.scenario", "result.score", "result.size_increase", "result.time_increase", "status"]))
# todos = [x['_id'] for x in a if len(x['result_size_increase']) == 0]
# for t in todo: os.system('''~/da/git/bin/exp.py -e with 'scenario = "{}"' '''.format(t.scenario.path))
# for t in todos: print(t); list_all({"config.scenario": t})[-1]._add_oh()
# # some non-existing ones were pop()ped from todos
# db.runs.update({_id: 23}, {$set: {"stabus": "FABLED"}})

# ### result with closest ...
# b = [x for x in list_all() if x.scenario.name == '0.22' and x.size_overhead]
# min(b, key=lambda x: abs(size - x.size_overhead))
# c = [x for x in list_all() if x.scenario.name == '0.22' and x.time_overhead]
# min(c, key=lambda x: abs(42 - x.time_overhead))
# d = [x for x in list_all() if x.scenario.name == '0.22']
# min(d, key=lambda x: abs(0.63 - x.score))
# min(d, key=lambda x: abs(0.26 - x.score))

# ### scatter plot of accuracy vs overhead
# SIZE = 30
# import results, mplot
# b = [x for x in results.list_all() if (x.scenario.name == 'new defense' or 'defense-client-nodelay' in x.scenario or 'disabled' in x.scenario) and x.scenario.num_sites == SIZE]
# plot = mplot.accuracy_vs_overhead(b)
# ## plot through
# d30 = pd.DataFrame([x.to_dict() for x in results.list_all() if x.scenario.num_sites == SIZE and '0.22' in x.scenario.name])
# a = d30[['size_overhead', 'score']]
# a.drop_duplicates(inplace=True) # and dropna()
# mod = lmfit.models.ExponentialModel()
# pars = mod.guess(a.score, x=a.size_overhead)
# out = mod.fit(a.score, pars, x=a.size_overhead)
# c = np.arange(plot.get_xbound()[1])
# plt.plot(b, [out.eval(x=x) for x in b], color=sns.color_palette("colorblind")[2])) #color: hack, but worked


# ### flavor comparison: no clear picture, but I seems better than II (bII fails)
# def color(pandas_result):
#     if 'defense-client' in pandas_result: return 'red'
#     if 'aII' in pandas_result: return 'yellow'
#     if 'aI' in pandas_result: return 'orange'
#     if 'bII' in pandas_result: return 'blue'
#     if 'bI' in pandas_result: return 'green'
#     return 'grey'
# c['color'] = c['scenario'].map(color)

# ### mongodb
# ## update element
# db.runs.update({"_id" : 359}, {$unset: {'result.size_increase': 0}})
# ## remove element (id found through result._id, this had wrong scenario name)
# db.runs.remove({_id: 486})

# ### CREATE RESULTS ON DUCKSTEIN
# ## list from [[file:scenario.py::##%20scenarios%20without%20result]]
# mkreik@duckstein:~$ for i in 'disabled/bridge--2017-10-08' ... 'defense-client-nodelay/bridge--2017-09-25'; do exp.py with scenario=$i; done

# ### rough plot all of size 30
# import pandas as pd
# import results
# d30 = pd.DataFrame([x.__dict__ for x in results.list_all() if x.scenario.num_sites == 30])
# d30.plot.scatter('size_overhead', 'score')

# ### list all scenarios without results
# a = {s: len(list(results.for_scenario(s))) for s in scenario.list_all()}
# [name for (name, count) in a.iteritems() if count == 0]

# ### rename scenario in db (old in git/data/skip/dump_before_scenario_rename)
# a = results.list_all()
# replace = [r for r in a if r.src['config']['scenario'].rstrip('/') != r.scenario.path]
# for c in replace: c.update({"config.scenario": c.scenario.path})
# for e in [r for r in a if 'llama' in r.src['config']['scenario'].lower()]:
#     e.update({"config.scenario": e.src['config']['scenario'].replace("llama", "defense-client")})
# ## by hand in db
# db.runs.update({_id: 55}, {$set: {"config.scenario": "disabled/06-09-2016--10@30"}})
# db.runs.updateMany({"config.scenario": "disabled/06-09@10"}, {$set: {"config.scenario": "disabled/06-09-2016--10@30"}})
