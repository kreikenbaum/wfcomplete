#!/usr/bin/env python
'''Analyses (Panchenko's) features returned from Counter class'''
import datetime
import doctest
import logging
import sys
import time
from types import NoneType

import numpy as np
from sklearn import cross_validation, ensemble, multiclass, neighbors, svm, tree

import counter
import fit
import plot_data
import scenario

LOGFORMAT = '%(levelname)s:%(filename)s:%(lineno)d:%(message)s'
#LOGLEVEL = logging.DEBUG
LOGLEVEL = logging.INFO
#LOGLEVEL = logging.WARN
TIME_SEPARATOR = '@'

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

## td: if ever used, have a look at _scale (needs to reset SCALER)
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
    clf.fit(fit._scale(X, clf), y)
    (X_test, y_test, y_testd) = counter.to_features_cumul(test)
    X_test = fit._scale(X_test, clf)
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
    uses cross_validation.train_test_split
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
    (train_ids, test_ids) = cross_validation.train_test_split(
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
    X1, Xtmp, y1, ytmp = cross_validation.train_test_split( #pylint: disable=invalid-name
        X, y, train_size=1. / 3, stratify=y)
    X2, X3, y2, y3 = cross_validation.train_test_split( #pylint: disable=invalid-name
        Xtmp, ytmp, train_size=.5, stratify=ytmp)
    return (X1, X2, X3, y1, y2, y3)


def _verbose_test_11(X, y, clf):
    '''cross-test (1) estimator on (1) X, y, print results and estimator name'''
    now = time.time()
    print _clf_params(clf),
    res = fit._eval(X, y, clf)
    print res.mean()
    logging.info('time: %s', time.time() - now)
    logging.debug('res: %s', res)


def _xtest(X_train, y_train, X_test, y_test, clf):
    '''cross_tests with estimator'''
    clf.fit(fit._scale(X_train, clf), y_train)
    return clf.score(fit._scale(X_test, clf), y_test)


def class_stats_to_table(class_stats):
    '''prints table from data in class_stats (gen_class_stats_list output)'''
    rows = class_stats[0].keys()
    rows.remove('id')
    cols = [j['id'] for j in class_stats]
    print '| |',
    for col in cols:
        print '{} |'.format(col),
    print ''
    for row in rows:
        print '| {}'.format(row),
        for col in class_stats:
            print '| {}'.format(col[row]),
        print '|'


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
    print '10-fold result: {}'.format(clf.best_score_)
    return clf


BGS = ["background--2016-08-17", "background--2016-11-18",
       "background--2016-11-22"]
def _add_background(foreground, name=None, background=None):
    '''@returns a combined instance with background set merged in'''
    if name:
        date = scenario.Scenario(name).date
        nextbg = min(BGS,
                     key=lambda x: abs(scenario.Scenario(x).date - date))
        background = counter.all_from_dir(nextbg)
    foreground['background'] = background['background']
    return foreground


def open_world(scenario_obj, y_bound=0.05):
    '''open-world (SVM) test on data, optimized on bounded auc.

    :return: (fpr, tpr, optimal_clf, roc_plot_mpl)'''
    # _train combines training and testing data and _test is grid-validation
    X, y, _ = scenario_obj.to_features_cumul()
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, train_size=2. / 3, stratify=y)
    result = fit.my_grid(X_train, y_train, auc_bound=y_bound)#, n_jobs=1)
#    clf = fit.sci_grid(X_train, y_train, c=2**15, gamma=2**-45,
#                   grid_args={"scoring": scorer})
    fpr, tpr, _, prob = fit.roc(result.clf, X_train, y_train, X_test, y_test)
    print 'bounded auc: {} (C: {}, gamma: {})'.format(
        fit.bounded_auc_score(result.clf, X_test, y_test, 0.01),
        result.clf.estimator.C, result.clf.estimator.gamma)
    return (fpr, tpr, result, plot_data.roc(fpr, tpr), prob)


def closed_world(scenarios, def0, cumul=True, with_svm=True, common=False):
    '''cross test on dirs: 1st has training data, rest have test

    :param: scenarios contains scenario dirs like sys.argv.  If
            scenarios has only one set, it is cross-validated etc.  If
            there are more than one, the first is taken as baseline
            and training, while the others are tested against this.
            As example, search for sys.argv = ... lines below
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
    print 'cross-validation on X,y'
    for clf in clfs:
        _verbose_test_11(X, y, clf)

    # vs test sets
    its_traces0 = scenarios[def0]
    for (scenario, its_traces) in scenarios.iteritems():
        if scenario == def0:
            continue
        print '\ntrain: {} VS {} (overhead {}%)'.format(
            def0, scenario, _size_increase(stats[def0], stats[scenario]))
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
            print '{}: {}'.format(_clf_name(clf), _xtest(X, y, X2, y2, clf)),
            print '({} seconds)'.format(time.time() - now)


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
        for scenario in scenarios:
            res = _misclassification_rates(
                scenarios[scenario0], scenarios[scenario], clf=clf)
            res['id'] = '{} with {}'.format(scenario, _clf_name(clf))
            out.append(res)
    return out


def outlier_removal_levels(scenario, clf=None):
    '''tests different outlier removal schemes and levels

    @param scenario: one "set" of data {site_1: traces, ..., site_n: traces}
    # outlier removal on both at the same time
    '''
    print 'combined outlier removal'
    for lvl in [1, 2, 3]:
        scenario_with_or = counter.outlier_removal(scenario, lvl)
        (train, test) = _tts(scenario_with_or)
        (X, y, _) = counter.to_features_cumul(train)
        if clf is None:
            clf = fit.my_grid(X, y)
        (X, y, _) = counter.to_features_cumul(test)
        print "level: {}".format(lvl)
        _verbose_test_11(X, y, clf)
    (train, test) = _tts(scenario)

    # separate outlier removal on train and test set
    print 'separate outlier removal for training and test data'
    for train_lvl in [1, 2, 3]:
        for test_lvl in [-1, 1, 2, 3]:
            (X, y, _) = counter.to_features_cumul(
                counter.outlier_removal(train, train_lvl))
            if isinstance(clf, NoneType):
                clf = fit.my_grid(X, y)
            (X, y, _) = counter.to_features_cumul(
                counter.outlier_removal(test, test_lvl))
            print "level train: {}, test: {}".format(train_lvl, test_lvl)
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
    scenarios = [scenario.Scenario(x) for x in argv[1:]]
    if 'background' in scenarios.values()[0].path:
        if len(scenarios) > 1:
            logging.warn('only first scenario chosen for open world analysis')
        return open_world(scenarios[0])
    else:
        return closed_world([x.get_traces() for x in scenarios],
                            argv[1], with_svm=with_svm, cumul=cumul)

# pylint: disable=line-too-long
# OLDER DATA (without bridge)
# sys.argv = ['', 'disabled/05-12@10']
# next: traces in between
# sys.argv = ['', 'disabled/06-09@10', '0.18.2/json-10/a-i-noburst', '0.18.2/json-10/a-ii-noburst', '0.15.3/json-10/cache', '0.15.3/json-10/nocache']
# sys.argv = ['', 'disabled/06-17@10-from', '0.18.2/json-10/a-i-noburst', '0.18.2/json-10/a-ii-noburst', '0.15.3/json-10/cache', '0.15.3/json-10/nocache'] #older
# missing:
# sys.argv = ['', 'disabled/06-17@10-from', 'retro/0', 'retro/1', 'retro/10', 'retro/20', 'retro/30', 'retro/5', '0.15.3/json-10/0', '0.15.3/json-10/1', '0.15.3/json-10/10', '0.15.3/json-10/20', '0.15.3/json-10/30', '0.15.3/json-10/40', '0.15.3/json-10/5', '0.19/0-ai', '0.19/0-bii', '0.19/20-bi', '0.19/20-bii', '0.19/aii-factor=0', '0.21']
# sys.argv = ['', 'disabled/2016-06-30', 'retro/0', 'retro/1', 'retro/10',
# 'retro/20', 'retro/30', 'retro/5', '0.15.3/json-10/0',
# '0.15.3/json-10/1', '0.15.3/json-10/10', '0.15.3/json-10/20',
# '0.15.3/json-10/30', '0.15.3/json-10/40', '0.15.3/json-10/5',
# '0.19/0-ai', '0.19/0-bii', '0.19/20-bi', '0.19/20-bii',
# '0.19/aii-factor=0', '0.21']

# sys.argv = ['', 'disabled/wtf-pad', 'wtf-pad']
# sys.argv = ['', 'disabled/06-17@100/', '0.18.2/json-100/b-i-noburst']
# sys.argv = ['', 'disabled/06-17@10-from', '0.20/0-ai', '0.20/0-bi',
# '0.20/20-ai', '0.20/20-bi', '0.20/40-bi', '0.20/0-aii', '0.20/0-bii',
# '0.20/20-aii', '0.20/20-bii', '0.20/40-aii', '0.20/40-bii']

# CLASSIFICATION RESULTS PER CLASS

# some_30 = top_30(means)
# timing = {k: _average_duration(v) for (k,v) in scenarios.iteritems()}
# outlier_removal_levels(scenarios[sys.argv[1]]) #td: try out

# PANCHENKO_PATH = os.path.join('..', 'sw', 'p', 'foreground-data', 'output-tcp')
# PANCHENKO_30 = os.path.join('..', 'sw', 'p', 'subsets', '30', 'foreground-data', 'output-tcp')
# PANCHENKO_bg = os.path.join('..', 'sw', 'p', 'subsets', 'background-1200', 'output-tcp')
# PANCHENKO_100 = os.path.join('..', 'sw', 'p', 'subsets', '30', 'foreground-data', 'output-tcp')
# PANCHENKO_bg2 = os.path.join('..', 'sw', 'p', 'subsets', 'background-4000', 'output-tcp')
# traces = counter.all_from_panchenko(PANCHENKO_PATH)

# variants
# RETRO
# ['retro/bridge/100--2016-09-15', 'retro/bridge/50--2016-09-16']
# ['retro/bridge/200--2016-10-02/', 'retro/bridge/200--2016-10-02-with-errs/']
# MAIN 0.22
#['0.22/10aI--2016-07-08', '0.22/30aI--2016-07-13', '0.22/50aI--2016-07-13', '0.22/5aII--2016-07-18', '0.22/5aI--2016-07-19', '0.22/10-maybe-aI--2016-07-23', '0.22/5aI--2016-07-25', '0.22/30aI--2016-07-25', '0.22/50aI--2016-07-26', '0.22/2aI--2016-07-23', '0.22/5aI--2016-08-26', '0.22/5aII--2016-08-25', '0.22/5bI--2016-08-27', '0.22/5bII--2016-08-27', '0.22/20aI--2016-09-10', '0.22/20aII--2016-09-10', '0.22/20bII--2016-09-12', '0.22/20bI--2016-09-13']
# SIMPLE
#['simple1/50', 'simple2/30', 'simple2/30-burst', 'simple1/10', 'simple2/5--2016-07-17', 'simple2/20']

# COMPLETE
# 07-06
# sys.argv = ['', 'disabled/bridge--2016-07-06', 'retro/bridge/30', 'retro/bridge/70', 'retro/bridge/50']
# sys.argv = ['', 'disabled/bridge--2016-07-06', 'simple1/50', 'simple2/30', 'simple2/30-burst', 'simple1/10', 'simple2/20']
# sys.argv = ['', 'disabled/bridge--2016-07-06', 'wtf-pad/bridge--2016-07-05', 'tamaraw']
# sys.argv = ['', 'disabled/bridge--2016-07-06', '0.22/10aI', '0.22/5aI--2016-07-19', '0.22/5aII--2016-07-18', '0.22/2aI--2016-07-23']
# sys.argv = ['', 'disabled/bridge--2016-07-06', '0.22/10aI--2016-07-08/', 'wtf-pad/bridge--2016-07-05', '0.22/30aI--2016-07-13/', '0.22/50aI--2016-07-13/']
# 07-21
# sys.argv = ['', 'disabled/bridge--2016-07-21', 'simple2/5--2016-07-17', '0.22/5aII--2016-07-18/', '0.22/5aI--2016-07-19/', '0.22/10-maybe-aI--2016-07-23/', '0.22/2aI--2016-07-23/', '0.22/30aI--2016-07-25/', '0.22/50aI--2016-07-26/', '0.22/5aI--2016-07-25/', '0.15.3/bridge']
# 08-14/15
# sys.argv = ['', 'disabled/bridge--2016-08-14', 'disabled/bridge--2016-08-15']
# 08-29 (also just FLAVORS)
# sys.argv = ['', 'disabled/bridge--2016-08-29', '0.22/5aI--2016-08-26', '0.22/5aII--2016-08-25', '0.22/5bI--2016-08-27', '0.22/5bII--2016-08-27']
# 09-09 (also just FLAVORS)
# sys.argv = ['', 'disabled/bridge--2016-09-09', '0.22/20aI--2016-09-10', '0.22/20aII--2016-09-10', '0.22/20bI--2016-09-13', '0.22/20bII--2016-09-12']
# 09-18 (also just RETRO)
# sys.argv = ['', 'disabled/bridge--2016-09-18', 'retro/bridge/100--2016-09-15', 'retro/bridge/50--2016-09-16']
#'disabled/bridge--2016-09-30'
# 09-23 (also just SIMPLE)
# sys.argv = ['', 'disabled/bridge--2016-09-21-100', 'simple2/5--2016-09-23-100/']
# sys.argv = ['', 'disabled/bridge--2016-09-26-100', 'simple2/5--2016-09-23-100/']
# sys.argv = ['', 'disabled/bridge--2016-09-26-100-with-errs', 'simple2/5--2016-09-23-100/']
# 10-06 (also just FLAVORS)
# sys.argv = ['', "disabled/bridge--2016-10-06-with-errors",
# "0.22/20aI--2016-10-07", "0.22/20aI--2016-10-07-with-errors",
# "0.22/20aII--2016-10-07", "0.22/20aII--2016-10-07-with-errors",
# "0.22/20bI--2016-10-08", "0.22/20bI--2016-10-08-with-errors",
# "0.22/20bII--2016-10-08", "0.22/20bII--2016-10-08-with-errors",
# "0.22/5aI--2016-10-09", "0.22/5aI--2016-10-09-with-errors",
# "0.22/5aII--2016-10-09", "0.22/5aII--2016-10-09-with-errors",
# "0.22/5bI--2016-10-10", "0.22/5bI--2016-10-10-with-errors",
# "0.22/5bII--2016-10-10", "0.22/5bII--2016-10-10-with-errors"]

# DISABLED
# 30
# sys.argv = ['', 'disabled/bridge--2016-07-06', 'disabled/bridge--2016-07-21', 'disabled/bridge--2016-08-14', 'disabled/bridge--2016-08-15', 'disabled/bridge--2016-08-29', 'disabled/bridge--2016-09-09', 'disabled/bridge--2016-09-18', 'disabled/bridge--2016-09-30', "disabled/bridge--2016-10-06-with-errors", "disabled/bridge--2016-10-16", "disabled/bridge--2016-10-16-with-errors"]
# 100
# sys.argv = ['', 'disabled/bridge--2016-08-30-100',
# 'disabled/bridge--2016-09-21-100', 'disabled/bridge--2016-09-26-100',
# 'disabled/bridge--2016-09-26-100-with-errs']

# NOT SO NEW
# sys.argv = ['', './disabled/bridge--2016-11-21']
# sys.argv = ['', './disabled/bridge--2016-11-04-100@50', './0.22/10aI--2016-11-04-50-of-100', './disabled/bridge--2016-11-27']
# NEWER
# christmas
# wtf-pad 2017


# TOP
# sys.argv = ['', 'disabled/bridge--2016-07-21', 'simple2/5--2016-07-17', '0.22/5aI--2016-07-19']
# sys.argv = ['', 'disabled/bridge--2016-07-06', 'wtf-pad/bridge--2016-07-05']

# disabled/p-foreground-data/30/output-tcp


# if by hand: change to the right directory before importing
# import os; os.chdir(os.path.join(os.path.expanduser('~') , 'da', 'git', 'data')); _=os.nice(20); sys.path.append(os.path.join(os.path.expanduser('~') , 'da', 'git', 'bin')); logging.basicConfig(format=LOGFORMAT, level=LOGLEVEL)

# pylint: enable=line-too-long
doctest.testmod()

if __name__ == "__main__":
    logging.basicConfig(format=LOGFORMAT, level=LOGLEVEL)
    main(sys.argv)
