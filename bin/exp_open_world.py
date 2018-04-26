#! /usr/bin/env python
'''runs open world experiment'''
import logging
import os
import random

import pymongo
from sacred import Experiment
from sacred.observers import MongoObserver

import analyse
import config
import scenario as scenario_module  # "scenario" name already used

ex = Experiment('wf_open_world')
ex.observers.append(MongoObserver.create())


# pylint: disable=unused-variable
@ex.config
def my_config():
    scenario = random.choice(scenario_module.list_all()).path
    or_level = None
    remove_small = None
    auc_bound = None
    background_size = 'auto'  # 'auto', number, None
    binarize = True
    exclude_sites = []
    current_sites = True  # fix e.g. google duplicates in bg set
# pylint: enable=unused-variable


# code duplication exp.py
@ex.capture
def run_exp(scenario, remove_small, or_level, auc_bound,
            background_size, binarize, exclude_sites, current_sites,
            _rnd):
    config.OR_LEVEL = config.OR_LEVEL if or_level is None else or_level
    config.REMOVE_SMALL = (config.REMOVE_SMALL if remove_small is None
                           else remove_small)
    scenario_obj = scenario_module.Scenario(scenario)
    (tpr, fpr, auroc, C, gamma, acc, y, yp, yd) = analyse.simulated_open_world(
        scenario_obj, auc_bound=auc_bound, binary=binarize,
        bg_size=background_size, exclude_sites=exclude_sites,
        current_sites=current_sites)
    return {
        'C': C,
        'gamma': gamma,
        'sites': scenario_obj.get_traces().keys(),
        'score': acc,
        'type': "accuracy",
        'outlier_removal': scenario_obj.trace_args,
        'size_increase': scenario_obj.size_increase(),
        'time_increase': scenario_obj.time_increase(),
        'fpr': fpr,
        'tpr': tpr,
        'auroc': auroc,
        'y_true': y.tolist(),
        'y_prediction': yp.tolist(),
        'y_domains': yd
    }


def _format_results(results):
    out = []
    for ((C, gamma), score) in results.iteritems():
        out.append([C, gamma, score])
    return out


@ex.automain
def my_main(scenario):
    _ = os.nice(20)
    db = pymongo.MongoClient().sacred
    if scenario in db.runs.distinct(
            "config.scenario",
            {"status": "COMPLETED",
             "experiment.mainfile": "exp_open_world.py"}):
        logging.warn("scenario already in database")
    return run_exp()  # pylint: disable=no-value-for-parameter

# ## inspect database:
# use sacred; db.runs.aggregate([{$match: {"$and": [{"config.scenario": {"$exists": 1}}, {"result.score": {"$exists": 1}}]}}, {$project: {"config.scenario": 1, "result.score": 1}},{$group: {_id: "$config.scenario", "score": {$max: "$result.score"}}}])
# ## check duplicates
# db.runs.aggregate([{"$match": {"$and": [
#         {"config.scenario": {"$exists": 1}},
#         {"result.score": {"$exists": 1}}]}},
#     {"$project": {"config.scenario": 1, "result.score": 1}},
#     {"$group": {
#         "_id": "$config.scenario",
#         "score": {"$push": "$result.score"},
#         "count": {"$sum": 1}}}])
