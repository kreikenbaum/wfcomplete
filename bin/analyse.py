#!/usr/bin/env python
'''Analyses (Panchenko's) features returned from Counter class'''
from __future__ import print_function
import doctest
import logging
import sys
import time
from types import NoneType

import numpy as np
from sklearn import ensemble, metrics, model_selection, multiclass
from sklearn import neighbors, preprocessing, svm, tree

import config
import counter
import fit
import results
#import mplot
import scenario

#TIME_SEPARATOR = '@'

# classifiers
GOOD = [ensemble.ExtraTreesClassifier(n_estimators=250),
        ensemble.RandomForestClassifier(),
        neighbors.KNeighborsClassifier(),
        tree.DecisionTreeClassifier()]
ALL = GOOD[:]
ALL.extend([ensemble.AdaBoostClassifier(),
            svm.SVC(gamma=2**-4)])
SVC_TTS_MAP = {}
ALL_MAP = {}

# td: check if correct to use like this (or rather like _size_increase)
# td: think if remove (not used)
def _average_duration(trace_dict):
    '''@return the average duration over all traces'''
    mean_std = _times_mean_std(trace_dict)
    return np.mean([x[0] for x in mean_std.values()])


def _class_predictions(cls, cls_predict):
    ''':returns: list: for each class in cls: what was it predicted to be'''
    out = [[] for _ in range(cls[-1] + 1)] # different empty arrays
    for (idx, elem) in enumerate(cls):
        out[elem].append(cls_predict[idx])
    return out


def _clf(**svm_params):
    '''@return default classifier with additional params'''
    return multiclass.OneVsRestClassifier(
        svm.SVC(class_weight="balanced", **svm_params))


def _clf_name(clf):
    '''@return name of estimator class'''
    return str(clf.__class__).split('.')[-1].split("'")[0]


def _clf_params(clf):
    '''@return name + params if SVM'''
    if 'SVC' in str(clf):
        try:
            return '{}_{}'.format(_clf_name(clf), clf.best_params_)
        except AttributeError:
            return '{}_{}'.format(_clf_name(clf), (clf.estimator.C,
                                                   clf.estimator.gamma))
    else:
        return _clf_name(clf)

## td: if ever used, have a look at scale (needs to reset SCALER)
# def _compare(X, y, X2, y2, clfs=GOOD):
#     for clf in clfs:
#         fit._eval(X, y, clf)
#         fit._eval(X2, y2, clf)

def _dict_elementwise(func, dict_1, dict_2):
    ''':return: {k: func(dict_1, dict_2)} for all keys k in dict_1'''
    return {k: func(dict_1[k], dict_2[k]) for k in dict_1}


def _find_domain(mean_per_dir, mean):
    '''@return (first) domain name with mean'''
    for place_means in mean_per_dir.values():
        for (domain, domain_mean) in place_means.items():
            if domain_mean == mean:
                return domain

# td: ref: traces == trace_list?


def _format_row(row):
    '''format row of orgtable to only contain relevant data (for tex export)'''
    out = [row[0]]
    out.extend(el[:6] for el in row[1:])
    return out


def _gen_url_list(y, y_domains):
    '''@return list of urls, the index is the class'''
    out = []
    for _ in range(y[-1] + 1):
        out.append([])
    for (idx, cls) in enumerate(y):
        if not out[cls]:
            out[cls] = y_domains[idx]
        else:
            assert out[cls] == y_domains[idx]
    return out


def _mean(trace_dict):
    '''@return a dict of {domain1: mean1, ... domainN: meanN}
    >>> _mean({'yahoo.com': [counter._test(3)]})
    {'yahoo.com': 1800.0}
    '''
    out = {}
    for (domain, trace_list) in trace_dict.iteritems():
        total = [i.get_total_both() for i in trace_list]
        out[domain] = np.mean(total)
    return out


def _misclassification_rates(train, test, clf=GOOD[0]):
    '''@return (mis-)classification rates per class in test'''
    (X, y, _) = counter.to_features_cumul(counter.outlier_removal(train))
    clf.fit(fit.scale(X, clf), y)
    (X_test, y_test, y_testd) = counter.to_features_cumul(test)
    X_test = fit.scale(X_test, clf)
    return _predict_percentages(
        _class_predictions(y_test, clf.predict(X_test)),
        _gen_url_list(y_test, y_testd))


def _predict_percentages(class_predictions_list, url_list):
    '''@return percentages how often a class was mapped to itself'''
    import collections
    out = {}
    for (idx, elem) in enumerate(class_predictions_list):
        out[url_list[idx]] = float(collections.Counter(elem)[idx]) / len(elem)
    return out


def _std(trace_dict):
    '''@return a dict of {domain1: std1, ... domainN: stdN}
    >>> _std({'yahoo.com': [counter._test(3)]})
    {'yahoo.com': 0.0}
    '''
    out = {}
    for (domain, trace_list) in trace_dict.iteritems():
        total = [i.get_total_both() for i in trace_list]
        out[domain] = np.std(total)
    return out


def _times_mean_std(trace_dict):
    '''analyse timing data (time overhead)

    @return a dict of {domain1: (mean1,std1)}, ... domainN: (meanN, stdN)}
    with mean and standard of timing data
    '''
    out = {}
    for (domain, trace_list) in trace_dict.iteritems():
        total = [i.timing[-1][0] for i in trace_list]
        out[domain] = (np.mean(total), np.std(total))
    return out


def _tts(trace_dict, test_size=1.0 / 3):
    '''train-test-split: splits trace_dict in train_dict and test_dict

    test_size = deep_len(test)/deep_len(train+test)
    uses model_selection.train_test_split
    @return (train_dict, test_dict) which together yield trace_dict
    >>> len(_tts({'yahoo.com': map(counter._test, [3,3,3])})[0]['yahoo.com'])
    2
    >>> len(_tts({'yahoo.com': map(counter._test, [3,3,3])})[1]['yahoo.com'])
    1
    '''
    ids = []
    for url in trace_dict:
        for i in range(len(trace_dict[url])):
            ids.append((url, i))
    (train_ids, test_ids) = model_selection.train_test_split(
        ids, test_size=test_size)
    train = {}
    test = {}
    for url in trace_dict:
        train[url] = []
        test[url] = []
    for (url, index) in train_ids:
        train[url].append(trace_dict[url][index])
    for (url, index) in test_ids:
        test[url].append(trace_dict[url][index])
    return (train, test)


def _tvts(X, y):
    '''@return X1, X2, X3, y1, y2, y3 with each 1/3 of the data (train,
validate, test)
    >> _tvts([[1], [1], [1], [2], [2], [2]], [1, 1, 1, 2, 2, 2])
    ([[1], [2]], [[1], [2]], [[1], [2]], [1, 2], [1, 2], [1, 2]) # modulo order
    '''
    X1, Xtmp, y1, ytmp = model_selection.train_test_split( #pylint: disable=invalid-name
        X, y, train_size=1. / 3, stratify=y)
    X2, X3, y2, y3 = model_selection.train_test_split( #pylint: disable=invalid-name
        Xtmp, ytmp, train_size=.5, stratify=ytmp)
    return (X1, X2, X3, y1, y2, y3)


def _verbose_test_11(X, y, clf):
    '''cross-test (1) estimator on (1) X, y, print results and estimator name'''
    now = time.time()
    print(_clf_params(clf), end='')
    res = fit._eval(X, y, clf)
    print(res.mean(), end='')
    logging.info('time: %s', time.time() - now)
    logging.debug('res: %s', res)


def _xtest(X_train, y_train, X_test, y_test, clf):
    '''cross_tests with estimator'''
    clf.fit(fit.scale(X_train, clf), y_train)
    return clf.score(fit.scale(X_test, clf), y_test)


#    p_ = clf.fit(X1, y1).predict_proba(X2)
#    return bounded_auc(y2, p_[:, 1], y_bound, pos_label=0)
def bounded_auc(y_true, y_predict, y_bound, **kwargs):
    '''@return bounded auc of (probabilistic) fitted classifier on data.'''
    newfpr, newtpr = _bounded_roc(y_true, y_predict, y_bound, **kwargs)
    return metrics.auc(newfpr, newtpr)


def bounded_roc(y_true, y_predict, y_bound, **kwargs):
    '''@return (fpr, tpr) within fpr-bounds'''
    assert 0 <= y_bound <= 1
    if y_predict.shape[1] == 2:
        y_predict = y_predict[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_true, y_predict, **kwargs)
    # plot_data.roc(fpr, tpr).savefig('/tmp/roc.pdf')
    # plt.close()
    newfpr = [x for x in fpr if x < y_bound]
    newfpr.append(y_bound)
    newtpr = np.interp(newfpr, fpr, tpr)
    return (newfpr, newtpr)


def bounded_auc_score(clf, X, y, y_bound=0.01):
    '''@return cross-validated bounded auc of clf on X and y'''
    scorer = metrics.make_scorer(
        _bounded_auc, needs_proba=True, y_bound=y_bound)
    y = list(_binarize(y, transform_to=1))
    return 1/y_bound * model_selection.cross_val_score(
        clf, X, y, cv=config.FOLDS, n_jobs=config.JOBS_NUM,
        scoring=scorer).mean()


## todo: maybe use prettytable
def class_stats_to_table(class_stats):
    '''prints table from data in class_stats (gen_class_stats_list output)'''
    rows = class_stats[0].keys()
    rows.remove('id')
    cols = [j['id'] for j in class_stats]
    print('| |', end='')
    for col in cols:
        print('{} |'.format(col), end='')
    print('')
    for row in rows:
        print('| {}'.format(row), end='')
        for col in class_stats:
            print('| {}'.format(col[row]), end='')
        print('|')


def compare_stats(scenario_names):
    '''@return a dict {scenario1: {domain1: {...}, ..., domainN: {...}},
    scenario2:..., ..., scenarioN: ...} with domain mean, standard distribution
    and labels'''
    scenario_dicts = counter.for_scenarios(scenario_names)
    means = {k: _mean(v) for (k, v) in scenario_dicts.iteritems()}
    stds = {k: _std(v) for (k, v) in scenario_dicts.iteritems()}
    out = []
    for scenario_name in scenario_names:
        logging.info('version: %s', scenario_name)
        default = {"plugin-version": scenario_name,
                   "plugin-enabled": 'disabled' not in scenario_name}
        for site in scenario_dicts[scenario_name]:
            tmp = dict(default)
            tmp['website'] = site
            tmp['mean'] = means[scenario_name][site]
            tmp['std'] = stds[scenario_name][site]
            out.append(tmp)
    return out


def simulated_original(traces, name=None):
    '''simulates original panchenko: does 10-fold cv on _all_ data, just
picks best result'''
    if name is not None and name in ALL_MAP:
        clf = ALL_MAP[name]
    else:
        clf = fit.helper(counter.outlier_removal(traces, 2),
                         cumul=True, folds=10)
        ALL_MAP[name] = clf
    print('10-fold result: {}'.format(clf.best_score_))
    return clf


def simulated_open_world(scenario_obj, auc_bound=0.1, binarize=False,
                         bg_size="auto", previous=False):
    '''@return (metrics for open world experiment
2. predict values
   - take best clf, predict
3. compute fpr and tpr (can use sklearn for this)
4. return all necessary values
   - tpr
   - fpr
     - multi- and single class
   - C
   - gamma
   - ?bounded auc?
   - ?auc?
   - accuracy
'''
    ow = scenario_obj.get_open_world()
    if binarize: ow = ow.binarize()
    X, y, domains = ow.get_features_cumul()
    X = preprocessing.MinMaxScaler().fit_transform(X) # scaling is idempotent
    if previous:
        result = max(results.for_scenario(a), key=lambda x: x.cumul)
        C = result.c
        gamma = result.gamma
        accuracy = result.cumul
    else:
        (clf_noprob, accuracy, _) = fit.my_grid(X, y) # auto scales
        C = clf_noprob.estimator.C
        gamma=clf_noprob.estimator.gamma
    clf = _clf(C=C, gamma=gamma)
    y_pred = model_selection.cross_val_predict(clf, X, y, cv=config.FOLDS,
                                               n_jobs=config.JOBS_NUM)
    confmat = metrics.confusion_matrix(y, y_pred)
    (tpr, fpr) = tpr_fpr(confmat)
    if binarize: # can (easily) compute auroc
        clf = _clf(C=C, gamma=gamma, probability=True)
        y_predprob = model_selection.cross_val_predict(clf, X, y,
                                                       cv=config.FOLDS,
                                                       n_jobs=config.JOBS_NUM,
                                                       method="predict_proba")
        fpr_array, tpr_array, _ = metrics.roc_curve(y, y_predprob)
        auroc = metrics.auc(fpr_array, tpr_array)
    else: auroc = None
    return (tpr, fpr, auroc, C, gamma, accuracy)


# due to https://stackoverflow.com/questions/31324218
def tpr_fpr(confusion_matrix):
    '''@return array of (tpr, fpr) pairs'''
    TP = np.diag(confusion_matrix) *1.0
    FP = confusion_matrix.sum(axis=0) - TP
    FN = confusion_matrix.sum(axis=1) - TP
    TN = confusion_matrix.sum() - (FP + FN + TP)
    return zip(TP/(TP+FN), FP/(FP+TN))


def open_world(scenario_obj, y_bound=0.05):
    '''open-world (SVM) test on data, optimized on bounded auc.

    :return: (fpr, tpr, optimal_clf, probabilities)'''
    # _train combines training and testing data and _test is grid-validation
    if 'background' not in scenario_obj.get_traces():
        scenario_obj = scenario_obj.get_open_world()
    X, y, _ = scenario_obj.binarize().get_features_cumul()
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, train_size=2. / 3, stratify=y)
    result = fit.my_grid(X_train, y_train, auc_bound=y_bound)#, n_jobs=1)
#    clf = fit.sci_grid(X_train, y_train, c=2**15, gamma=2**-45,
#                   grid_args={"scoring": scorer})
    fpr, tpr, _, prob = fit.roc(result.clf, X_train, y_train, X_test, y_test)
    print('{}-bounded auc: {} (C: {}, gamma: {})'.format(
        y_bound,
        fit.bounded_auc_score(result.clf, X_test, y_test, 0.01),
        result.clf.estimator.C, result.clf.estimator.gamma))
    return (fpr, tpr, result, prob)
#    return (fpr, tpr, result, mplot.roc(fpr, tpr), prob)


def closed_world(scenarios, def0, cumul=True, with_svm=True, common=False):
    '''cross test on dirs: 1st has training data, rest have test

    :param: scenarios: dict name : traces

        If scenarios has only one set, it is cross-validated etc.  If
        there are more than one, the first is taken as baseline and
        training, while the others are tested against this.

    :param: cumul triggers CUMUL, else version 1,
    :param: common determines whether to reduce the test data to
            common keys.
    '''
    stats = {k: scenario._bytes_mean_std(v) for (k, v) in scenarios.iteritems()}
    # durations = {k: _average_duration(v) for (k,v) in scenarios.iteritems()}

    # no-split, best result of 10-fold tts
    simulated_original(scenarios[def0], def0)

    # training set
    (train, test) = _tts(scenarios[def0])
    clfs = GOOD[:]
    if with_svm:
        if def0 in SVC_TTS_MAP and cumul:
            logging.info('reused svc: %s for scenario: %s',
                         SVC_TTS_MAP[def0],
                         def0)
            clfs.append(SVC_TTS_MAP[def0])
        else:
            now = time.time()
            (clf, _, _) = fit.helper(
                counter.outlier_removal(train, 2), cumul)
            logging.debug('parameter search took: %s', time.time() - now)
            if cumul:
                SVC_TTS_MAP[def0] = clf
                clfs.append(SVC_TTS_MAP[def0])
            else:
                clfs.append(clf)

    # X,y for eval
    if cumul:
        (X, y, _) = counter.to_features_cumul(counter.outlier_removal(test, 1))
    else:
        (X, y, _) = counter.to_features(counter.outlier_removal(test, 1))
    # evaluate accuracy on all of unaddoned
    print('cross-validation on X,y')
    for clf in clfs:
        _verbose_test_11(X, y, clf)

    # vs test sets
    its_traces0 = scenarios[def0]
    for (scenario_path, its_traces) in scenarios.iteritems():
        if scenario_path == def0:
            continue
        print('\ntrain: {} VS {} (overhead {}%)'.format(
            def0, scenario_path,
            _size_increase(stats[def0], stats[scenario_path])))
        if common and its_traces.keys() != its_traces0.keys():
            # td: refactor code duplication with above (search for keys = ...)
            keys = set(its_traces0.keys())
            keys = keys.intersection(its_traces.keys())
            tmp = {}
            tmp0 = {}
            for key in keys:
                tmp0[key] = its_traces0[key]
                tmp[key] = its_traces[key]
            its_traces0 = tmp0
            its_traces = tmp
        if cumul:
            (X2, y2, _) = counter.to_features_cumul(its_traces)
        else:
            max_len = _dict_elementwise(
                max,
                counter._find_max_lengths(its_traces0),
                counter._find_max_lengths(its_traces))
            (X, y, _) = counter.to_features(
                counter.outlier_removal(its_traces0, 2), max_len)
            (X2, y2, _) = counter.to_features(
                counter.outlier_removal(its_traces, 1), max_len)
        for clf in clfs:
            now = time.time()
            print('{}: {}'.format(_clf_name(clf), _xtest(X, y, X2, y2, clf)),
                  end='')
            print('({} seconds)'.format(time.time() - now))


def gen_class_stats_list(scenarios,
                         scenario0='auto',
                         clfs=[GOOD[0]]):
    '''@return list of _misclassification_rates() with scenario name'''
    if scenario0 == 'auto':
        scenario0 = [x for x in scenarios if 'disabled' in x][0]
        if scenario0 in SVC_TTS_MAP:
            clfs.append(SVC_TTS_MAP[scenario0])
    out = []
    for clf in clfs:
        for scenario_path in scenarios:
            res = _misclassification_rates(
                scenarios[scenario0], scenarios[scenario_path], clf=clf)
            res['id'] = '{} with {}'.format(scenario_path, _clf_name(clf))
            out.append(res)
    return out


def outlier_removal_levels(scenario_path, clf=None):
    '''tests different outlier removal schemes and levels

    @param scenario_path: one "set" of data {site_1: traces, ..., site_n: traces}
    # outlier removal on both at the same time
    '''
    print('combined outlier removal')
    for lvl in [1, 2, 3]:
        scenario_with_or = counter.outlier_removal(scenario_path, lvl)
        (train, test) = _tts(scenario_with_or)
        (X, y, _) = counter.to_features_cumul(train)
        if clf is None:
            clf = fit.my_grid(X, y)
        (X, y, _) = counter.to_features_cumul(test)
        print("level: {}".format(lvl))
        _verbose_test_11(X, y, clf)
    (train, test) = _tts(scenario_path)

    # separate outlier removal on train and test set
    print('separate outlier removal for training and test data')
    for train_lvl in [1, 2, 3]:
        for test_lvl in [-1, 1, 2, 3]:
            (X, y, _) = counter.to_features_cumul(
                counter.outlier_removal(train, train_lvl))
            if isinstance(clf, NoneType):
                clf = fit.my_grid(X, y)
            (X, y, _) = counter.to_features_cumul(
                counter.outlier_removal(test, test_lvl))
            print("level train: {}, test: {}".format(train_lvl, test_lvl))
            _verbose_test_11(X, y, clf)


def top_30(mean_per_dir):
    '''@return 30 domains with well-interspersed trace means sizes

    @param is f.ex. means from compare_stats above.'''
    all_means = []
    for p_means in mean_per_dir.values():
        all_means.extend(p_means.values())
    percentiles = np.percentile(all_means,
                                np.linspace(0, 100, 31),
                                interpolation='lower')
    out = set()
    for mean in percentiles:
        out.add(_find_domain(mean_per_dir, mean))
    return out


def main(argv, with_svm=True, cumul=True):
    '''loads stuff, triggers either open or closed-world eval'''
    if len(argv) == 1:
        argv.append('.')
    # by hand: scenarios = counter.for_scenarios(sys.argv[1:])
    scenarios = [scenario.Scenario(x, smart=True) for x in argv[1:]]
    if 'background' in scenarios[0].get_traces():
        if len(scenarios) > 1:
            logging.warn('only first scenario chosen for open world analysis')
        return open_world(scenarios[0])
    else:
        closed_world({x.path: x.get_traces() for x in scenarios},
                     scenarios[0].path, with_svm=with_svm, cumul=cumul)


# pylint: disable=line-too-long
# ====== BY HAND ========
# import os; import logging; os.chdir(os.path.join(os.path.expanduser('~') , 'da', 'git', 'data')); _=os.nice(20); sys.path.append(os.path.join(os.path.expanduser('~') , 'da', 'git', 'bin')); logging.basicConfig(format=LOGFORMAT, level=LOGLEVEL)
# pylint: enable=line-too-long
doctest.testmod()

if __name__ == "__main__":
    os.nice(20)
    main(sys.argv)
