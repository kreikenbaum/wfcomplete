#!/usr/bin/env python
'''Analyses (Panchenko's) features returned from Counter class'''

import numpy as np
from sklearn import cross_validation, ensemble, multiclass, neighbors, preprocessing, svm, tree
from scipy.stats.mstats import gmean
import doctest
import logging
import sys
import time

import counter

JOBS_NUM = -3 # 1 (version1 to 4-6 (cumul onduckstein)), maybe also -4, ...
# panchenko grid takes about 16% mem/cpu: 3-4 should be fine, 5 ok'ish, 6 limit
LOGFORMAT='%(levelname)s:%(filename)s:%(lineno)d:%(message)s'
#LOGLEVEL = logging.DEBUG
#LOGLEVEL = logging.INFO
LOGLEVEL = logging.WARN
TIME_SEPARATOR = '@'

### classifiers
GOOD = [ensemble.ExtraTreesClassifier(n_estimators=250),
        ensemble.RandomForestClassifier(),
        neighbors.KNeighborsClassifier(),
        tree.DecisionTreeClassifier()]
ALL = GOOD[:]
ALL.extend([ensemble.AdaBoostClassifier(),
            svm.SVC(gamma=2**-4)])
SVC_TTS_MAP = {}
ALL_MAP = {}

scaler = None

# td: check if correct to use like this (or rather like _size_increase)
def _average_duration(counter_dict):
    '''@return the average duration over all traces'''
    ms = times_mean_std(counter_dict)
    return np.mean([x[0] for x in ms.values()])

def _bytes_mean_std(counter_dict):
    '''@return a dict of {domain1: (mean1,std1}, ... domainN: (meanN, stdN)}
    >>> _bytes_mean_std({'yahoo.com': [counter._test(3)]})
    {'yahoo.com': (1800.0, 0.0)}
    '''
    out = {}
    for (domain, counter_list) in counter_dict.iteritems():
        total = [counter.get_total_in() for counter in counter_list]
        out[domain] = (np.mean(total), np.std(total))
    return out

def _class_predictions(y2, y2_predict):
    '''@return list: for each class in y2: what was it predicted to be'''
    out = []
    for i in range(y2[-1]+1):
        out.append([])
    for (idx, elem) in enumerate(y2):
        out[elem].append(y2_predict[idx])
    return out

def _clf_name(clf):
    '''@return name of estimator class'''
    return str(clf.__class__).split('.')[-1].split("'")[0]

def _clf_params(clf):
    '''@return name + params if SVM'''
    if  'SVC' in str(clf):
        return ('{}(C={}, gamma={})'.format(_clf_name(clf),
                                           clf.estimator.C,
                                           clf.estimator.gamma))
    else:
        return _clf_name(clf)

# td: if ever used, have a look at _scale (needs to reset SCALER)
def _compare(X, y, X2, y2, clfs=GOOD):
    for clf in clfs:
        _test(X, y, clf)
        _test(X2, y2, clf)

# courtesy of http://stackoverflow.com/a/38060351
def _dict_elementwise(func, d1, d2):
    return {k: func(d1[k], d2[k]) for k in d1}

def _find_domain(mean_per_dir, mean):
    '''@return (first) domain name with mean'''
    for place_means in mean_per_dir.values():
        for (domain, domain_mean) in place_means.items():
            if domain_mean == mean:
                return domain

def _find_max_lengths(counters):
    '''determines maximum lengths of variable-length features'''
    max_lengths = counters.values()[0][0].variable_lengths()
    all_lengths = []
    for domain, domain_values in counters.iteritems():
        for trace in domain_values:
            all_lengths.append(trace.variable_lengths())
    for lengths in all_lengths:
        for key in lengths.keys():
            if max_lengths[key] < lengths[key]:
                max_lengths[key] = lengths[key]
    return max_lengths

def _format_row(row):
    '''format row of orgtable to only contain relevant data (for tex export)'''
    out = [row[0]]
    out.extend(el[:6] for el in row[1:])
    return out

def _gen_url_list(y, y_domains):
    '''@return list of urls, the index is the class'''
    out = []
    for i in range(y[-1]+1):
        out.append([])
    for (idx, cls) in enumerate(y):
        if not out[cls]:
            out[cls] = y_domains[idx]
        else:
            assert out[cls] == y_domains[idx]
    return out

def _mean(counter_dict):
    '''@return a dict of {domain1: mean1, ... domainN: meanN}
    >>> _mean({'yahoo.com': [counter._test(3)]})
    {'yahoo.com': 1800.0}
    '''
    out = {}
    for (domain, counter_list) in counter_dict.iteritems():
        total = [counter.get_total_both() for counter in counter_list]
        out[domain] = np.mean(total)
    return out

# td: simple doctest/unit test
def _misclassification_rates(train, test, clf=GOOD[0]):
    '''@return (mis-)classification rates per class in test'''
    (X, y, y_d) = to_features_cumul(counter.outlier_removal(train))
    clf.fit(_scale(X, clf), y)
    (X2, y2, y2d) = to_features_cumul(test)
    X2 = _scale(X2, clf)
    return _predict_percentages(_class_predictions(y2, clf.predict(X2)),
                                _gen_url_list(y2, y2d))

def _my_grid(X, y,
             cs=np.logspace(11, 17, 4, base=2),
             gammas=np.logspace(-3, 3, 4, base=2),
             results={}, num_jobs=JOBS_NUM, folds=5):
    '''grid-search on fixed params

    @param results are previously computed results {(c, g): accuracy, ...}
    @return tuple (optimal_classifier, results_object)'''
    best = None
    bestres = 0
    for c in cs:
        for g in gammas:
            clf = multiclass.OneVsRestClassifier(svm.SVC(
                gamma=g, C=c, class_weight='balanced'))
            if (c,g) in results:
                current = results[(c,g)]
            else:
                current = _test(X, y, clf, num_jobs, folds=folds)
                results[(c, g)] = current
            if not best or bestres.mean() < current.mean():
                best = clf
                bestres = current
            logging.debug('c: {:8} g: {:10} acc: {}'.format(c, g,
                                                            current.mean()))
    if (best.estimator.C in (cs[0], cs[-1])
        or best.estimator.gamma in (gammas[0], gammas[-1])):
        logging.warn('optimal parameters found at the border. c:%f, g:%f',
                     best.estimator.C, best.estimator.gamma)
        return _my_grid(X, y,
                       _new_search_range(best.estimator.C),
                       _new_search_range(best.estimator.gamma),
                       results)
    else:
        logging.info('grid result: {}'.format(best))
        return best, results

def _my_grid_helper(counter_dict, outlier_removal=True, nj=JOBS_NUM,
                    cumul=True, folds=5):
    '''@return grid-search on counter_dict result (clf, results)'''
    if outlier_removal:
        counter_dict = counter.outlier_removal(counter_dict)
    if cumul:
        (X, y, _) = to_features_cumul(counter_dict)
        return _my_grid(X, y, num_jobs=nj, folds=folds)
    else: # panchenko 1
        (X, y, _) = to_features(counter_dict)
        return _my_grid(X, y, num_jobs=nj,
                        cs=np.logspace(15, 19, 3, base=2),
                        gammas=np.logspace(-17, -21, 3, base=2))

def _new_search_range(best_param):
    '''returns new array of parameters to search in

    (use if best result was at border)
    '''
    return [best_param / 2, best_param, best_param * 2]

def _predict_percentages(class_predictions_list, url_list):
    '''@return percentages how often a class was mapped to itself'''
    import collections
    out = {}
    for (idx, elem) in enumerate(class_predictions_list):
        out[url_list[idx]] =  float(collections.Counter(elem)[idx])/len(elem)
    return out

def _std(counter_dict):
    '''@return a dict of {domain1: std1, ... domainN: stdN}
    >>> _std({'yahoo.com': [counter._test(3)]})
    {'yahoo.com': 0.0}
    '''
    out = {}
    for (domain, counter_list) in counter_dict.iteritems():
        total = [counter.get_total_both() for counter in counter_list]
        out[domain] = np.std(total)
    return out

def _scale(X, clf):
    '''assumption: svc never called on two different data sets in
    sequence.  That is: _scale(X_train, svc), _scale(X_test, svc),
    _scale(Y_train, svc), _scale(Y_test, svc), will not
    happen. (without a _scale(..., non_svc) in between). The first is
    treated as training data, the next as test.

    @return scaled X if estimator is SVM, else just X

    '''
    global scaler
    if 'SVC' in str(clf):
        logging.debug("_scaled on svc %s", _clf_name(clf))
        if not scaler:
            scaler = preprocessing.MinMaxScaler()
            return scaler.fit_transform(X)
        else:
            return scaler.transform(X)
    else:
        scaler = None
        return X

def _test(X, y, clf, nj=JOBS_NUM, folds=5):
    '''tests estimator with X, y, @return result (ndarray)'''
    X = _scale(X, clf)
    return cross_validation.cross_val_score(clf, X, y, cv=folds, n_jobs=nj)

# unused, but could be useful
def _times_mean_std(counter_dict):
    '''analyse timing data (time overhead)

    @return a dict of {domain1: (mean1,std1)}, ... domainN: (meanN, stdN)}
    with mean and standard of timing data
    '''
    out = {}
    for (domain, counter_list) in counter_dict.iteritems():
        total = [counter.timing[-1][0] for counter in counter_list]
        out[domain] = (np.mean(total), np.std(total))
    return out

def _verbose_test_11(X, y, clf):
    '''cross-test (1) estimator on (1) X, y, print results and estimator name'''
    t = time.time()
    scale = True if 'SVC' in str(clf) else False
    print _clf_params(clf),
    res = _test(X, y, clf)
    print res.mean()
    logging.info('time: %s', time.time() - t)
    logging.debug('res: %s', res)

def _xtest(X_train, y_train, X_test, y_test, clf):
    '''cross_tests with estimator'''
    clf.fit(_scale(X_train, clf), y_train)
    return clf.score(_scale(X_test, clf), y_test)

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

def compare_stats(dirs):
    '''@return a dict {dir1: {domain1: {...}, ..., domainN: {...}},
    dir2:..., ..., dirN: ...} with domain mean, standard distribution
    and labels'''
    defenses = counter.for_defenses(dirs)
    means = {k: _mean(v) for (k,v) in defenses.iteritems()}
    stds = {k: _std(v) for (k,v) in defenses.iteritems()}
    out = []
    for d in dirs:
        logging.info('version: %s', d)
        el = {"plugin-version": d,
              "plugin-enabled": False if 'disabled' in d else True}
        for site in defenses[d]:
            tmp = dict(el)
            tmp['website'] = site
            tmp['mean'] = means[d][site]
            tmp['std'] = stds[d][site]
            out.append(tmp)
    return out

def simulated_original(counters, name=None, folds=10):
    '''simulates original panchenko: does 10-fold cv on _all_ data, just
picks best result'''
    if name is not None and name in ALL_MAP:
        res = ALL_MAP[name]
    else:
        _,res = _my_grid_helper(counter.outlier_removal(counters, 2),
                                cumul=True, folds=folds)
        ALL_MAP[name] = res
    print '10-fold result: {}'.format(max(map(np.mean, res.values())))

def _size_increase_helper(two_defenses):
    return _size_increase(two_defenses[two_defenses.keys()[0]],
                          two_defenses[two_defenses.keys()[1]])

def _size_increase(base, compare):
    '''@return how much bigger/smaller is =compare= than =base= (in %)'''
    diff = {}
    if base.keys() != compare.keys():
        keys = set(base.keys())
        keys = keys.intersection(compare.keys())
        logging.warn("keys are different, just used {} common keys"
                     .format(len(keys)))
    else:
        keys = base.keys()
    for k in keys:
        diff[k] = float(compare[k][0]) / base[k][0]
    return 100 * (gmean(diff.values()) -1)

def size_test(argv, outlier_removal=True):
    '''1. collect traces
    2. create stats
    3. evaluate for each vs first'''
    defenses = counter.for_defenses(argv[1:])
    stats = {k: _bytes_mean_std(v) for (k,v) in defenses.iteritems()}
    defense0 = argv[1]
    for d in argv[2:]:
        print '{}: {}'.format(d, _size_increase(stats[defense0], stats[d]))

def cross_test(argv, cumul=True, with_svm=False, num_jobs=JOBS_NUM, cc=True):
    '''cross test on dirs: 1st has training data, rest have test

    argv is like sys.argv, cumul triggers CUMUL, else version 1'''
    # call with 1: x-validate test that
    # call with 2+: also train 1 (split), test 2,3,4,...
    defenses = counter.for_defenses(argv[1:])
    stats = {k: _bytes_mean_std(v) for (k,v) in defenses.iteritems()}
    # domain), compare
    # durations = {k: _average_duration(v) for (k,v) in defenses.iteritems()}

    defense0 = argv[1] if len(argv) > 1 else '.'

    # no-split, best result of 10-fold tts
    simulated_original(defenses[defense0], defense0)
    # clf,res = _my_grid_helper(counter.outlier_removal(defenses[defense0], 2),
    #                           cumul, folds=10)
    # print '10-fold result: {}'.format(max(map(np.mean, res.values())))

    # training set
    (train, test) = tts(defenses[defense0])
    CLFS = GOOD[:]
    if with_svm:
        if defense0 in SVC_TTS_MAP and cumul:
            logging.info('reused svc: %s for defense: %s',
                         SVC_TTS_MAP[defense0],
                         defense0) #debug?
            CLFS.append(SVC_TTS_MAP[defense0])
        else:
            t = time.time()
            clf,_ = _my_grid_helper(counter.outlier_removal(train, 2), cumul)
            logging.debug('parameter search took: %s', time.time() -t)
            if cumul:
                SVC_TTS_MAP[defense0] = clf
                CLFS.append(SVC_TTS_MAP[defense0])
            else:
                CLFS.append(clf)

    # X,y for eval
    if cumul:
        (X, y, _) = to_features_cumul(counter.outlier_removal(test, 1))
    else:
        (X, y, _) = to_features(counter.outlier_removal(test, 1))
    # evaluate accuracy on all of unaddoned
    print 'cross-validation on X,y'
    for clf in CLFS:
        _verbose_test_11(X, y, clf)

    # vs test sets
    its_counters0 = defenses[defense0]
    for (defense, its_counters) in defenses.iteritems():
        if defense == defense0:
            continue
        print '\ntrain: {} VS {} (overhead {}%)'.format(
            defense0, defense, _size_increase(stats[defense0], stats[defense]))
        if cc and its_counters.keys() != its_counters0.keys():
            # td: refactor code duplication with above (search for keys = ...)
            keys = set(its_counters0.keys())
            keys = keys.intersection(its_counters.keys())
            tmp = {}; tmp0 = {}
            for key in keys:
                tmp0[key] = its_counters0[key]
                tmp[key] = its_counters[key]
            its_counters0 = tmp0
            its_counters = tmp
            logging.warn("keys are different, just used {} common keys"
                         .format(len(keys)))

        if cumul:
            (X2, y2, _) = to_features_cumul(its_counters)
        else:
            l = _dict_elementwise(max,
                                  _find_max_lengths(its_counters0),
                                  _find_max_lengths(its_counters))
            (X, y, _) = to_features(counter.outlier_removal(its_counters0, 2),
                                    l)
            (X2, y2, _2) = to_features(counter.outlier_removal(its_counters, 1),
                                       l)
        for clf in CLFS:
            t = time.time()
            print '{}: {}'.format(_clf_name(clf), _xtest(X, y, X2, y2, clf)), 
            print '({} seconds)'.format(time.time() - t)

def gen_class_stats_list(defenses,
                         defense0='auto',
                         clfs=[GOOD[0]]):
    '''@return list of _misclassification_rates() with defense name'''
    if defense0 == 'auto':
        defense0 = [x for x in defenses if 'disabled' in x][0]
        if defense0 in SVC_TTS_MAP:
            clfs.append(SVC_TTS_MAP[defense0])
    out = []
    for clf in clfs:
        for c in defenses:
            res = _misclassification_rates(defenses[defense0], defenses[c], clf=clf)
            res['id'] = '{} with {}'.format(c, _clf_name(clf))
            out.append(res)
    return out

def outlier_removal_levels(defense, clf=None):
    '''tests different outlier removal schemes and levels

    SET EITHER DEFENSE XOR TRAIN_TEST
    @param defense: one "set" of data {site_1: counters, ..., site_n: counters}
    @param train_test: tuple of two "sets" (train, test) each like defense'''
    # outlier removal on both at the same time
    print 'combined outlier removal'
    for lvl in [1,2,3]:
        defense_with_or = counter.outlier_removal(defense, lvl)
        (train, test) = tts(defense_with_or)
        (X, y, _) = to_features_cumul(train)
        if type(clf) is type(None):
            clf,_ = _my_grid(X, y)
        (X, y, _) = to_features_cumul(test)
        print "level: {}".format(lvl)
        _verbose_test_11(X, y, clf)
    (train, test) = tts(defense)

    # separate outlier removal on train and test set
    print 'separate outlier removal for training and test data'
    for train_lvl in [1,2,3]:
        for test_lvl in [-1,1,2,3]:
            (X, y, _) = to_features_cumul(counter.outlier_removal(train,
                                                                  train_lvl))
            if type(clf) is type(None):
                clf,_ = _my_grid(X, y)
            (X, y, _) = to_features_cumul(counter.outlier_removal(test,
                                                                  test_lvl))
            print "level train: {}, test: {}".format(train_lvl, test_lvl)
            _verbose_test_11(X, y, clf)

def site_sizes(stats):
    '''@return {'url1': [size0, ..., sizeN-1], ..., urlM: [...]}

    stats = {k: _bytes_mean_std(v) for (k,v) in defenses.iteritems()}'''
    a = stats.keys()
    a.sort()
    out = {}
    for url in stats[a[0]].keys():
        out[url] = []
        for defense in a:
            out[url].append(stats[defense][url][0])
    return out

def to_features(counters, max_lengths=None):
    '''transforms counter data to panchenko.v1-feature vector pair (X,y)

    if max_lengths is given, use that instead of trying to determine'''
    if not max_lengths:
        max_lengths = _find_max_lengths(counters)

    X_in = []
    out_y = []
    class_number = 0
    domain_names = []
    for domain, dom_counters in counters.iteritems():
        for count in dom_counters:
            if not count.warned:
                X_in.append(count.panchenko(max_lengths))
                out_y.append(class_number)
                domain_names.append(domain)
            else:
                logging.warn('%s: one discarded', domain)
        class_number += 1
    return (np.array(X_in), np.array(out_y), domain_names)

# td: refactor: code duplication with to_features
def to_features_cumul(counters):
    '''transforms counter data to CUMUL-feature vector pair (X,y)'''
    X_in = []
    out_y = []
    class_number = 0
    domain_names = []
    for domain, dom_counters in counters.iteritems():
        for count in dom_counters:
            if not count.warned:
                X_in.append(count.cumul())
                out_y.append(class_number)
                domain_names.append(domain)
            else:
                logging.warn('%s: one discarded', domain)
        class_number += 1
    return (np.array(X_in), np.array(out_y), domain_names)


def to_libsvm(X, y, fname='libsvm_in'):
    """writes lines in X with labels in y to file 'libsvm_in'"""
    f = open(fname, 'w')
    for i, row in enumerate(X):
        f.write(str(y[i]))
        f.write(' ')
        for no, val in enumerate(row):
            if val != 0:
                f.write('{}:{} '.format(no+1, val))
        f.write('\n')

def top_30(mean_per_dir):
    '''@return 30 domains with well-interspersed trace means sizes

    @param is f.ex. means from compare_stats above.'''
    all_means = []
    for (defense, p_means) in mean_per_dir.items():
        all_means.extend(p_means.values())
    percentiles = np.percentile(all_means,
                                np.linspace(0, 100, 31),
                                interpolation='lower')
    out = set()
    for mean in percentiles:
        out.add(_find_domain(mean_per_dir, mean))
    return out

def tts(counter_dict, test_size=1.0/3):
    '''train-test-split: splits counter_dict in train_dict and test_dict

    test_size = deep_len(test)/deep_len(train)
    uses cross_validation.train_test_split
    @return (train_dict, test_dict) which together yield counter_dict
    >>> len(tts({'yahoo.com': map(counter._test, [3,3,3])})[0]['yahoo.com'])
    2
    >>> len(tts({'yahoo.com': map(counter._test, [3,3,3])})[1]['yahoo.com'])
    1
    '''
    ids = []
    for url in counter_dict:
        for i in range(len(counter_dict[url])):
            ids.append((url, i))
    (train_ids, test_ids) = cross_validation.train_test_split(
        ids, test_size=test_size)
    train = {}
    test = {}
    for url in counter_dict:
        train[url] = []
        test[url] = []
    for (url, index) in train_ids:
        train[url].append(counter_dict[url][index])
    for (url, index) in test_ids:
        test[url].append(counter_dict[url][index])
    return (train, test)

    #_test(X, y, svm.SVC(kernel='linear')) #problematic, but best
    ### random forest
    ## feature importance
    # forest = ensemble.ExtraTreesClassifier(n_estimators=250)
    # forest.fit(X, y)
    # forest.feature_importances_
    ### extratree param
    # for num in range(50, 400, 50):
    #     _test(X, y, ensemble.ExtraTreesClassifier(n_estimators=num))
    ### linear params
    # cstart, cstop = -5, 5
    # Cs = np.logspace(cstart, cstop, base=10, num=(abs(cstart - cstop)+1))
    # for c in Cs:
    #     _test(X, y, svm.SVC(C=c, kernel='linear'))
    ### metrics (single)
    # from scipy.spatial import distance
    # for dist in [distance.braycurtis, distance.canberra,
    #              distance.chebyshev, distance.cityblock, distance.correlation,
    #              distance.cosine, distance.euclidean, distance.sqeuclidean]:
    #     _test(X, y, neighbors.KNeighborsClassifier(metric='pyfunc', func=dist))3
    ### td: knn + levenshtein
    # import math
    # def mydist(x, y):
    #     fixedm = distance.sqeuclidean(x[:8], y[:8])
    #     xbounds1 = (10, 10+x[8])
    #     ybounds1 = (10, 10+y[8])
    #     variable1 = gdl.metric(x[xbounds1[0]:xbounds1[1]],
    #                            y[ybounds1[0]:ybounds1[1]])
    #     xbounds2 = (10+x[8], 10+x[8]+x[9])
    #     ybounds2 = (10+x[8], 10+x[8]+y[9])
    #     variable2 = gdl.metric(x[xbounds2[0]:xbounds2[1]],
    #                            y[ybounds2[0]:ybounds2[1]])
    #     return math.sqrt(fixedm + variable1 + variable2)
#    (X, y ,y_dom) = to_features_cumul(counters)

### OLDER DATA (without bridge)
# sys.argv = ['', 'disabled/06-09@10', '0.18.2/json-10/a_i_noburst', '0.18.2/json-10/a_ii_noburst', '0.15.3/json-10/cache', '0.15.3/json-10/nocache'] #older
# sys.argv = ['', 'disabled/wfpad', 'wfpad']
# sys.argv = ['', 'disabled/06-17@100/', '0.18.2/json-100/b_i_noburst']
# sys.argv = ['', 'disabled/06-17@10_from', '20.0/0_ai', '20.0/0_bi', '20.0/20_ai', '20.0/20_bi', '20.0/40_ai', '20.0/40_bi', '20.0/0_aii', '20.0/0_bii', '20.0/20_aii', '20.0/20_bii', '20.0/40_aii', '20.0/40_bii']

### CLASSIFICATION RESULTS PER CLASS

# defenses = counter.for_defenses(sys.argv[1:])
# some_30 = top_30(means)
# timing = {k: _average_duration(v) for (k,v) in defenses.iteritems()}
# outlier_removal_levels(defenses[sys.argv[1]]) #td: try out

# PANCHENKO_PATH = os.path.join('..', 'sw', 'p', 'foreground-data', 'output-tcp')
# counters = counter.all_from_panchenko(PANCHENKO_PATH)

## CREATE WANG' BATCH DIRECTORIES. call this in data/ directory
##   on update, alter [[diplomarbeit.org::*How to get Wang-kNN to work]]
# for root, dirs, files in os.walk('.'):
#     # plots or already processed
#     if (not re.search('/(plots|path|batch|results)', root) and
#         not dirs and files):
#         print root
#         counter.dir_to_wang(root, remove_small=False)

### variants
## RETRO
# ['0.15.3-retrofixed/bridge/100__2016_09_15', '0.15.3-retrofixed/bridge/50__2016_09_16']
# ['0.15.3-retrofixed/bridge/200__2016-10-02/', '0.15.3-retrofixed/bridge/200__2016-10-02_with_errs/']
## MAIN 0.22
#['0.22/10aI__2016-07-08', '0.22/30aI__2016-07-13', '0.22/50aI__2016-07-13', '0.22/5aII__2016-07-18', '0.22/5aI__2016-07-19', '0.22/10_maybe_aI__2016-07-23', '0.22/5aI__2016-07-25', '0.22/30aI__2016-07-25', '0.22/50aI__2016-07-26', '0.22/2aI__2016-07-23', '0.22/5aI__2016-08-26', '0.22/5aII__2016-08-25', '0.22/5bI__2016-08-27', '0.22/5bII__2016-08-27', '0.22/20aI__2016-09-10', '0.22/20aII__2016-09-10', '0.22/20bII__2016-09-12', '0.22/20bI__2016-09-13']
## SIMPLE
#['simple1/50', 'simple2/30', 'simple2/30-burst', 'simple1/10', 'simple2/5__2016-07-17', 'simple2/20']

# 07-06
# sys.argv = ['', 'disabled/bridge__2016-07-06', '0.15.3-retrofixed/bridge/30', '0.15.3-retrofixed/bridge/70', '0.15.3-retrofixed/bridge/50']
# sys.argv = ['', 'disabled/bridge__2016-07-06', 'simple1/50', 'simple2/30', 'simple2/30-burst', 'simple1/10', 'simple2/20']
# sys.argv = ['', 'disabled/bridge__2016-07-06', 'wfpad/bridge__2016-07-05', 'tamaraw']
# sys.argv = ['', 'disabled/bridge__2016-07-06', '0.22/10aI', '0.22/5aI__2016-07-19', '0.22/5aII__2016-07-18', '0.22/2aI__2016-07-23']
# sys.argv = ['', 'disabled/bridge__2016-07-06', '0.22/10aI__2016-07-08/', 'wfpad/bridge__2016-07-05', '0.22/30aI__2016-07-13/', '0.22/50aI__2016-07-13/']
# 07-21
# sys.argv = ['', 'disabled/bridge__2016-07-21', 'simple2/5__2016-07-17', '0.22/5aII__2016-07-18/', '0.22/5aI__2016-07-19/', '0.22/10_maybe_aI__2016-07-23/', '0.22/2aI__2016-07-23/', '0.22/30aI__2016-07-25/', '0.22/50aI__2016-07-26/', '0.22/5aI__2016-07-25/', '0.15.3/bridge']
# 08-14/15
# sys.argv = ['', 'disabled/bridge__2016-08-14', 'disabled/bridge__2016-08-15']
# 08-29 (also just FLAVORS)
# sys.argv = ['', 'disabled/bridge__2016-08-29', '0.22/5aI__2016-08-26', '0.22/5aII__2016-08-25', '0.22/5bI__2016-08-27', '0.22/5bII__2016-08-27']
# 09-09 (also just FLAVORS)
# sys.argv = ['', 'disabled/bridge__2016-09-09', '0.22/20aI__2016-09-10', '0.22/20aII__2016-09-10', '0.22/20bI__2016-09-13', '0.22/20bII__2016-09-12']
# 09-18 (also just RETRO)
# sys.argv = ['', 'disabled/bridge__2016-09-18', '0.15.3-retrofixed/bridge/100__2016_09_15', '0.15.3-retrofixed/bridge/50__2016_09_16']
#'disabled/bridge__2016-09-30'
# 09-23 (also just SIMPLE)
# sys.argv = ['', 'disabled/bridge__2016-09-21_100', 'simple2/5__2016-09-23_100/']
# sys.argv = ['', 'disabled/bridge__2016-09-26_100', 'simple2/5__2016-09-23_100/']
# sys.argv = ['', 'disabled/bridge__2016-09-26_100_with_errs', 'simple2/5__2016-09-23_100/']
# 10-06 (also just FLAVORS)
# sys.argv = ['', "disabled/bridge__2016-10-06_with_errors", "0.22/22@20aI__2016-10-07", "0.22/22@20aI__2016-10-07_with_errors", "0.22/22@20aII__2016-10-07", "0.22/22@20aII__2016-10-07_with_errors", "0.22/22@20bI__2016-10-08", "0.22/22@20bI__2016-10-08_with_errors", "0.22/22@20bII__2016-10-08", "0.22/22@20bII__2016-10-08_with_errors", "0.22/22@5aI__2016-10-09", "0.22/22@5aI__2016-10-09_with_errors", "0.22/22@5aII__2016-10-09", "0.22/22@5aII__2016-10-09_with_errors", "0.22/22@5bI__2016-10-10", "0.22/22@5bI__2016-10-10_with_errors", "0.22/22@5bII__2016-10-10", "0.22/22@5bII__2016-10-10_with_errors"]

### DISABLED
# 30
# sys.argv = ['', 'disabled/bridge__2016-07-06', 'disabled/bridge__2016-07-21', 'disabled/bridge__2016-08-14', 'disabled/bridge__2016-08-15', 'disabled/bridge__2016-08-29', 'disabled/bridge__2016-09-09', 'disabled/bridge__2016-09-18', 'disabled/bridge__2016-09-30', "disabled/bridge__2016-10-06_with_errors", "disabled/bridge__2016-10-16", "disabled/bridge__2016-10-16_with_errors"]
## 100
# sys.argv = ['', 'disabled/bridge__2016-08-30_100', 'disabled/bridge__2016-09-21_100', 'disabled/bridge__2016-09-26_100', 'disabled/bridge__2016-09-26_100_with_errs']

# NEW
# ['./disabled/bridge__2016-11-04_100@50_100_classes', '0.22/10aI__2016-11-04_50_of_100']
# './disabled/bridge__2016-11-04_100@50_subsets/bridge__2016-11-04_100@50_top35'



### TOP
# sys.argv = ['', 'disabled/bridge__2016-07-21', 'simple2/5__2016-07-17', '0.22/5aI__2016-07-19']
# sys.argv = ['', 'disabled/bridge__2016-07-06', 'wfpad/bridge__2016-07-05']

# if by hand: change to the right directory before importing
# import os; os.chdir(os.path.join(os.path.expanduser('~') , 'da', 'git', 'data'))
if __name__ == "__main__":
    doctest.testmod()
    logging.basicConfig(format=LOGFORMAT, level=LOGLEVEL)


    cross_test(sys.argv, with_svm=True) #, cumul=False)
