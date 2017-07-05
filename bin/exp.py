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

DIR = '/home/uni/da/git/data/'

ex = Experiment('wf_alternatives')
ex.observers.append(MongoObserver.create())


@ex.config
def my_config_cw():
    scenario = 'disabled/06-09@10'

    
@ex.capture
def run_exp(scenario, _rnd):
    counters = counter.for_defenses([os.path.join(DIR, scenario)]).values()[0]
    result = analyse.simulated_original(counters)
    import pdb; pdb.set_trace()
    return {
        'C': result.clf.estimator.C,
        'gamma': result.clf.estimator.gamma,
        'sites': counters.keys(),
        'score': result.best_score_,
        'C_gamma_result': _format_results(result.results)
    }


def _format_results(results):
    out = []
    for ((c, gamma), score) in results.iteritems():
        out.append([c, gamma, score])
    return out


@ex.automain
def my_main(scenario):
    db = pymongo.MongoClient().sacred
    if scenario in db.runs.distinct("config.scenario", {"status": "COMPLETED"}):
        raise Exception("scenario already in database")
    return run_exp()

# inspect database:
# db.runs.aggregate([{$match: {"$and": [{"config.scenario": {"$exists": 1}}, {"result.score": {"$exists": 1}}]}}, {$project: {"config.scenario": 1, "result.score": 1}},{$group: {_id: "$config.scenario", "score": {$max: "$result.score"}}}])
