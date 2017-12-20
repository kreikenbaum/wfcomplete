#! /usr/bin/env python
'''runs open world experiment, the "sacred" experimentation framework'''
import logging
import os


import pymongo
from sacred import Experiment
from sacred.observers import MongoObserver

import analyse
import config
import fit
from scenario import Scenario # "scenario" name already used

ex = Experiment('wf_open_world')
ex.observers.append(MongoObserver.create())


@ex.config
def my_config():
    scenario = 'disabled/06-09@10/'
    or_level = None
    remove_small = None
    auc_bound = 0.1
    background_size = 'auto' #, number, None
    binarize = False
    use_previous_cw = False

# code duplication exp.py
@ex.capture
def run_exp(scenario, remove_small, or_level, auc_bound,
            background_size, binarize, use_previous_cw, _rnd):
    config.OR_LEVEL = config.OR_LEVEL if or_level is None else or_level
    config.REMOVE_SMALL = config.REMOVE_SMALL if remove_small is None else remove_small
    scenario_obj = Scenario(scenario)
    (fpr, tpr, auroc, C, gamma, accuracy) = analyse.simulated_open_world(
        scenario_obj, auc_bound, binarize, background_size, use_previous_cw)
    return {
        'C': C,
        'gamma': gamma,
        'sites': scenario_obj.get_traces().keys(),
        'score': accuracy,
        'type': "accuracy",
        'outlier_removal': scenario_obj.trace_args,
        'size_increase': scenario_obj.size_increase(),
        'time_increase': scenario_obj.time_increase(),
        'fpr': fpr,
        'tpr': tpr,
        'auroc': auroc
    }


def _format_results(results):
    out = []
    for ((c, gamma), score) in results.iteritems():
        out.append([c, gamma, score])
    return out


@ex.automain
def my_main(scenario):
    _ = os.nice(20)
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
