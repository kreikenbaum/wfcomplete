#! /usr/bin/env python
import logging
import os


import numpy as np
import pymongo
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn import pipeline, cross_validation, model_selection, svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

import analyse
import counter
from scenario import Scenario, TRACE_ARGS # scenario name already used

DIR = '/home/uni/da/git/data/'

ex = Experiment('wf_alternatives')
ex.observers.append(MongoObserver.create())


@ex.config
def my_config_cw():
    scenario = 'disabled/06-09@10/'
    trace_args = TRACE_ARGS
#    scenario = Scenario('wtf-pad/bridge--2017-01-08')

    
@ex.capture
def run_exp(scenario, trace_args, _rnd):
    s = Scenario(scenario, trace_args)
    traces = s.get_traces()
    result = analyse.simulated_original(traces)
    return {
        'C': result.clf.estimator.C,
        'gamma': result.clf.estimator.gamma,
        'sites': traces.keys(),
        'score': result.best_score_,
        'C_gamma_result': _format_results(result.results),
        'outlier_removal': s.trace_args,
        'size_increase': s.size_increase()
    }


def _format_results(results):
    out = []
    for ((c, gamma), score) in results.iteritems():
        out.append([c, gamma, score])
    return out


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


@ex.automain
def my_main(scenario):
    _=os.nice(20)
    db = pymongo.MongoClient().sacred
    if scenario in db.runs.distinct("config.scenario", {"status": "COMPLETED"}):
        logging.warn("scenario already in database")
    return run_exp()

## inspect database:
# use sacred; db.runs.aggregate([{$match: {"$and": [{"config.scenario": {"$exists": 1}}, {"result.score": {"$exists": 1}}]}}, {$project: {"config.scenario": 1, "result.score": 1}},{$group: {_id: "$config.scenario", "score": {$max: "$result.score"}}}])
## check duplicates
# db.runs.aggregate([{$match: {"$and": [{"config.scenario": {"$exists": 1}}, {"result.score": {"$exists": 1}}]}}, {$project: {"config.scenario": 1, "result.score": 1}},{$group: {_id: "$config.scenario", "score": {$push: "$result.score"}, "count": {$sum: 1}}}])
