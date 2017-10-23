#! /usr/bin/env python
import logging
import os


import numpy as np
import pymongo
from sacred import Experiment
from sacred.observers import MongoObserver

import analyse
import config
import counter
from scenario import Scenario # "scenario" name already used

ex = Experiment('wf_open_world')
ex.observers.append(MongoObserver.create())


@ex.config
def my_config():
    scenario = 'disabled/06-09@10/'
    or_level = None
    remove_small = None
    auc_bound = 0.1
    background_size = None # 'auto', number

    
@ex.capture
def run_exp(scenario, remove_small, or_level, auc_bound, background_size, _rnd):
    config.OR_LEVEL = or_level or config.OR_LEVEL
    config.REMOVE_SMALL = remove_small or config.REMOVE_SMALL
    s = Scenario(scenario)
    X, y, d = a.get_open_world(background_size).get_features_cumul()
    result = my_grid(X, y, auc_bound=0.1)
    return {
        'C': result.clf.estimator.C,
        'gamma': result.clf.estimator.gamma,
        'sites': s.get_traces().keys(),
        'score': result.best_score_,
        'type': "cumul",
        'C_gamma_result': _format_results(result.results),
        'outlier_removal': s.trace_args,
        'size_increase': s.size_increase(),
        'time_increase': s.time_increase()
    }


def _format_results(results):
    out = []
    for ((c, gamma), score) in results.iteritems():
        out.append([c, gamma, score])
    return out


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
# db.runs.aggregate([{"$match": {"$and": [
#         {"config.scenario": {"$exists": 1}},
#         {"result.score": {"$exists": 1}}]}},
#     {"$project": {"config.scenario": 1, "result.score": 1}},
#     {"$group": {
#         "_id": "$config.scenario",
#         "score": {"$push": "$result.score"},
#         "count": {"$sum": 1}}}])
