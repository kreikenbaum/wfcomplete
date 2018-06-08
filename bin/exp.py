#! /usr/bin/env python
'''runs closed-world experiment'''
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

ex = Experiment('wf_alternatives')
ex.observers.append(MongoObserver.create())


@ex.config
def my_config():
    scenario = 'disabled/06-09@10/'
    or_level = None
    remove_small = None
    current_sites = False  # fix e.g. google duplicates in bg set

@ex.capture
def run_exp(scenario, remove_small, or_level, current_sites, _rnd):
    config.OR_LEVEL = config.OR_LEVEL if or_level is None else or_level
    config.REMOVE_SMALL = config.REMOVE_SMALL if remove_small is None else remove_small
    s = Scenario(scenario)
    traces = s.get_traces(current_sites=current_sites)
    result = analyse.simulated_original(traces)
    return {
        'C': result.clf.estimator.C,
        'gamma': result.clf.estimator.gamma,
        'sites': traces.keys(),
        'score': result.best_score_,
        'type': "cumul",
        'C_gamma_result': _format_results(result.results),
        # todo: remove duplicates (or_level...), after checking that they match
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
