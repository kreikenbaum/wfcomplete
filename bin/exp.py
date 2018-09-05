#! /usr/bin/env python
'''runs closed-world experiment'''
import json
import logging
import os
import tempfile

import numpy as np
import pymongo
from sacred import Experiment
from sacred.observers import MongoObserver

from sklearn import model_selection, preprocessing

import analyse
import config
import counter
from scenario import Scenario # "scenario" name already used

ex = Experiment('wf_alternatives')
ex.observers.append(MongoObserver.create())

def _add_as_artifact(element, name):
    '''add variables that are too big for sacred results'''
    filename = tempfile.mktemp()
    with open(filename, "w") as f:
        json.dump(element, f)
    ex.add_artifact(filename, name)


@ex.config
def my_config():
    scenario = 'disabled/06-09@10/'
    or_level = None
    remove_small = None
    current_sites = False  # fix e.g. google duplicates in bg set
    remove_timeout = None
    exclude_sites = []

@ex.capture
def run_exp(scenario, remove_small, or_level, current_sites, remove_timeout,
            exclude_sites, _rnd):
    config.OR_LEVEL = config.OR_LEVEL if or_level is None else or_level
    config.REMOVE_SMALL = (config.REMOVE_SMALL if remove_small is None
                           else remove_small)
    config.REMOVE_TIMEOUT = (config.REMOVE_TIMEOUT if remove_timeout is None
                             else remove_timeout)
    scenario_obj = Scenario(scenario, exclude_sites=exclude_sites)
    traces = scenario_obj.get_traces(current_sites=current_sites)
    result = analyse.simulated_original(traces)
    X, y, d = scenario_obj.get_features_cumul(current_sites)
    X = preprocessing.MinMaxScaler().fit_transform(X)  # scaling is idempotent
    clf = result.clf.estimator
    y_pred = model_selection.cross_val_predict(
        clf, X, y, cv=10, n_jobs=config.JOBS_NUM)
    _add_as_artifact(y.tolist(), "y_true")
    _add_as_artifact(y_pred.tolist(), "y_prediction")
    _add_as_artifact(d, "y_domains")
    return {
        'C': result.clf.estimator.C,
        'gamma': result.clf.estimator.gamma,
        'sites': traces.keys(),
        'score': result.best_score_,
        'type': "cumul",
        'C_gamma_result': _format_results(result.results),
        # todo: remove duplicates (or_level...), after checking that they match
        'outlier_removal': scenario_obj.trace_args,
        'size_increase': scenario_obj.size_increase(),
        'time_increase': scenario_obj.time_increase()
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
