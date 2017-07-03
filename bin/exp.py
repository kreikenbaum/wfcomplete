#! /usr/bin/env python
import os

import numpy as np
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
#    import pdb; pdb.set_trace()
    return {
        'C': result.clf.estimator.C,
        'gamma': result.clf.estimator.gamma,
        'sites': counters.keys(),
        'score': result.best_score_,
        'C_gamma_result': format_results(result.results)
    }


def _format_results(results):
    out = []
    for ((c, gamma), score) in results.iteritems():
        out.append([c, gamma, score])


@ex.automain
def my_main():
    return run_exp()
