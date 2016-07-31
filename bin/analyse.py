#!/usr/bin/env python
'''Analyses (Panchenko's) features returned from Counter class'''

import numpy as np
from sklearn import cross_validation, ensemble, multiclass, neighbors, svm, tree
import doctest
import logging
import sys

import counter

JOBS_NUM = 2 # 1 (version1 to 4-6 (cumul onduckstein)), maybe also -4, ...
# panchenko grid takes about 16% mem/cpu: 3-4 should be fine, 5 ok'ish, 6 limit
LOGFORMAT='%(levelname)s:%(filename)s:%(lineno)d:%(message)s'
#LOGLEVEL = logging.DEBUG
LOGLEVEL = logging.INFO
#LOGLEVEL = logging.WARN
TIME_SEPARATOR = '@'

### classifiers
GOOD = [ensemble.ExtraTreesClassifier(n_estimators=250),
        ensemble.RandomForestClassifier(),
        neighbors.KNeighborsClassifier(),
        tree.DecisionTreeClassifier()]
ALL = GOOD[:]
ALL.extend([ensemble.AdaBoostClassifier(),
            svm.SVC(gamma=2**-4)])

def _average_bytes(mean_std_dict):
    '''@return the average size over all traces'''
    return np.mean([x[0] for x in mean_std_dict.values()])

def _average_duration(counter_dict):
    '''@return the average duration over all traces'''
    ms = times_mean_std(counter_dict)
    return np.mean([x[0] for x in ms.values()])

def _bytes_mean_std(counter_dict):
    '''@return a dict of {domain1: (mean1,std1}, ... domainN: (meanN, stdN)}
    >>> _bytes_mean_std({'yahoo.com': [counter._ptest(3)]})
    {'yahoo.com': (1800.0, 0.0)}
    '''
    out = {}
    for (domain, counter_list) in counter_dict.iteritems():
        total = [counter.get_total_both() for counter in counter_list]
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
    >>> _mean({'yahoo.com': [counter._ptest(3)]})
    {'yahoo.com': 1800.0}
    '''
    out = {}
    for (domain, counter_list) in counter_dict.iteritems():
        total = [counter.get_total_both() for counter in counter_list]
        out[domain] = np.mean(total)
    return out


def _my_grid(X, y,
            cs=np.logspace(11, 17, 4, base=2),
            gammas=np.logspace(-3, 3, 4, base=2),
            results={}):
    '''grid-search on fixed params

    @param results are previously computed results {(c, g): accuracy, ...}
    @return optimal_classifier, results_object'''
    best = None
    bestres = 0
    for c in cs:
        for g in gammas:
            clf = multiclass.OneVsRestClassifier(svm.SVC(
                gamma=g, C=c, class_weight='balanced'))
            if (c,g) in results:
                current = results[(c,g)]
            else:
                current = _test(X, y, clf, scale=True)
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
        return best, results

def _my_grid_helper(counter_dict, cumul=True, outlier_removal=True):
    '''@return grid-search on counter_dict result (clf, results)'''
    if outlier_removal:
        counter_dict = counter.outlier_removal(counter_dict)
    if cumul:
        (X, y, _) = to_features_cumul(counter_dict)
    else:
        (X, y, _) = to_features(counter_dict)
    return _my_grid(X, y)

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
    >>> _std({'yahoo.com': [counter._ptest(3)]})
    {'yahoo.com': 0.0}
    '''
    out = {}
    for (domain, counter_list) in counter_dict.iteritems():
        total = [counter.get_total_both() for counter in counter_list]
        out[domain] = np.std(total)
    return out

def _scale(X, clf):
    '''@return scaled X if estimator is SVM, else just X'''
    if 'SVC' in str(clf):
        tmp = np.copy(X)
        with np.errstate(divide='ignore', invalid='ignore'):
            tmp /= np.max(np.abs(tmp), axis=0)
        return np.nan_to_num(tmp)
    else:
        return X

# td merge etc with _scale()
def _test(X, y, clf=GOOD[0], nj=JOBS_NUM, verbose=True, scale=False):
    '''tests estimator with X, y, @return result (ndarray)'''
    if scale:
        X = np.copy(X)
        with np.errstate(divide='ignore', invalid='ignore'):
            X /= np.max(np.abs(X), axis=0)
        X = np.nan_to_num(X)
    return cross_validation.cross_val_score(clf, X, y, cv=5, n_jobs=nj)

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
    scale = True if 'SVC' in str(clf) else False
    print _clf_name(clf),
    res = _test(X, y, clf, scale=scale)
    print '{}, {}'.format(res.mean(), res)

def _xtest(X_train, y_train, X_test, y_test, clf, scale=False):
    '''cross_tests with estimator'''
    if scale:
        X_train = np.copy(X_train)
        X_test = np.copy(X_test)
        with np.errstate(divide='ignore', invalid='ignore'):
            X_train /= np.max(np.abs(X_train), axis=0)
            X_test /= np.max(np.abs(X_test), axis=0)
        X_train = np.nan_to_num(X_train)
        X_test = np.nan_to_num(X_test)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)

def cross_test(argv, cumul=True, with_svm=False):
    '''cross test on dirs: 1st has training data, rest have test

    argv is like sys.argv, cumul triggers CUMUL, else version 1'''
    # call with 1: x-validate test that
    # call with 2+: also train 1 (split), test 2,3,4,...
    places = counter.Counter.for_places(argv[1:], False)
    stats = {k: _bytes_mean_std(v) for (k,v) in places.iteritems()}
    sizes = {k: _average_bytes(v) for (k,v) in stats.iteritems()}
    # td: continue here, recompute duration (was not averaged per
    # domain), compare
    # durations = {k: _average_duration(v) for (k,v) in places.iteritems()}

    place0 = argv[1] if len(argv) > 1 else '.'

    # training set
    (train, test) = tts(places[place0])
    if with_svm:
        if not "OneVsRestClassifier" in [_clf_name(clf) for clf in GOOD]:
            clf,_ = _my_grid_helper(train)
            print 'grid result: {}'.format(clf) # maybe disable in production
            GOOD.append(clf)
        else:
            logging.info("reused existing OVR-SVM-classifier")
    # X,y for eval
    if cumul:
        (X, y, _) = to_features_cumul(counter.outlier_removal(test, 1))
    else:
        (X, y, _) = to_features(counter.outlier_removal(test, 1))
    # evaluate accuracy on all of unaddoned
    print 'cross-validation on X,y'
    for clf in GOOD:
        _verbose_test_11(X, y, clf)

    # vs test sets
    for (place, its_counters) in places.iteritems():
        if place == place0:
            continue
        print '\ntrain: {} VS {} (overhead {}%)'.format(
            place0, place, 100.0*(sizes[place]/sizes[place0] -1))
        if cumul:
            (X2, y2, _) = to_features_cumul(its_counters)
        else:
            l = _dict_elementwise(max,
                                  _find_max_lengths(places[place0]),
                                  _find_max_lengths(its_counters))
            (X, y, _) = to_features(places[place0], l)
            (X2, y2, _2) = to_features(its_counters, l)

        for clf in GOOD:
            scale = True if 'SVC' in str(clf) else False
            print '{}: {}'.format(_clf_name(clf),
                                  _xtest(X, y, X2, y2, clf, scale=scale))

def compare_stats(dirs):
    '''@return a dict {dir1: {domain1: {...}, ..., domainN: {...}},
    dir2:..., ..., dirN: ...} with domain mean, standard distribution
    and labels
    '''
    places = counter.Counter.for_places(dirs, False)
    means = {k: _mean(v) for (k,v) in places.iteritems()}
    stds = {k: _std(v) for (k,v) in places.iteritems()}
    out = []
    for d in dirs:
        logging.info('version: %s', d)
        el = {"plugin-version": d,
              "plugin-enabled": False if 'disabled' in d else True}
        for site in places[d]:
            tmp = dict(el)
            tmp['website'] = site
            tmp['mean'] = means[d][site]
            tmp['std'] = stds[d][site]
            out.append(tmp)
    return out

# outdated, lacks correct OR etc
def cumul_vs_panchenko(counters):
    '''tests version1 and cumul on counters'''
    (X, y, y_domains) = to_features(counters)
    (X2, y2, y2_domains) = to_features_cumul(counters)
    _compare(X, y, X2, y2)

def gen_class_stats_list(places,
        compare=['disabled/bridge', 'wfpad/bridge', 'simple2/5', '0.22/5aI'],
        clfs=[GOOD[0]]):
    '''@return list of show_class_stats() output amended with defense name'''
    out = []
    for clf in clfs:
        for c in compare:
            res = show_class_stats(places[compare[0]], places[c], clf=clf)
            res['id'] = '{} with {}'.format(c, _clf_name(clf))
            out.append(res)
    return out

def outlier_removal_vs_without(counters):
    (X, y, y_domains) = to_features_cumul(counter.outlier_removal(counters))
    (X2, y2, y2_domains) = to_features_cumul(counters)
    _compare(X, y, X2, y2)

def show_class_stats(train, test, clf=GOOD[0]):
    '''@return (mis-)classification rates per class in test'''
    (X, y, y_d) = to_features_cumul(counter.outlier_removal(train))
    clf.fit(_scale(X, clf), y)
    (X2, y2, y2d) = to_features_cumul(test)
    X2 = _scale(X2, clf)
    return _predict_percentages(_class_predictions(y2, clf.predict(X2)),
                                _gen_url_list(y2, y2d))

def test_or(place):
    '''tests different outlier removal schemes and levels'''
    (train, test) = tts(place)
    for train_lvl in [1,2,3]:
        for test_lvl in [-1,1,2,3]:
            (X, y, _) = to_features_cumul(counter.outlier_removal(train,
                                                                  train_lvl))
            clf,_ = _my_grid(X, y)
            (X, y, _) = to_features_cumul(counter.outlier_removal(test,
                                                                  test_lvl))
            print "level train: {}, test: {}".format(train_lvl, test_lvl)
            _verbose_test_11(X, y, clf)

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
    for (place, p_means) in mean_per_dir.items():
        all_means.extend(p_means.values())
    percentiles = np.percentile(vals,
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
    >>> len(tts({'yahoo.com': map(counter._ptest, [3,3,3])})[0]['yahoo.com'])
    2
    >>> len(tts({'yahoo.com': map(counter._ptest, [3,3,3])})[1]['yahoo.com'])
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
    #grid rbf
#     cstart, cstop = -45, -35
#     Cs = np.logspace(cstart, cstop, base=10, num=(abs(cstart - cstop)+1))
# #    Gs = np.logspace(gstart, gstop, base=10, num=10*(abs(gstart - gstop)+1))
#     gamma = 4.175318936560409e-10
#     for c in Cs:
# #        for gamma in Gs:
#         _test(X, y, svm.SVC(C=c, gamma=gamma))
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
    # sys.argv = ['', 'disabled/06-09@10', '0.18.2/json-10/a_i_noburst', '0.18.2/json-10/a_ii_noburst', '0.15.3/json-10/cache', '0.15.3/json-10/nocache'] #older
    # sys.argv = ['', 'disabled/wfpad', 'wfpad']
#    (X, y ,y_dom) = to_features_cumul(counters)
#_test(X, y, nj=1)

# older data (without bridge)
    # sys.argv = ['', 'disabled/06-17@100/', '0.18.2/json-100/b_i_noburst']
    # sys.argv = ['', 'disabled/06-17@10_from', '20.0/0_ai', '20.0/0_bi', '20.0/20_ai', '20.0/20_bi', '20.0/40_ai', '20.0/40_bi', '20.0/0_aii', '20.0/0_bii', '20.0/20_aii', '20.0/20_bii', '20.0/40_aii', '20.0/40_bii']

# defenses = counter.Counter.for_places(sys.argv[1:], False)
# some_30 = top_30(means)
# timing = {k: _average_duration(v) for (k,v) in defenses.iteritems()}

if __name__ == "__main__":
    doctest.testmod()
    logging.basicConfig(format=LOGFORMAT, level=LOGLEVEL)

    # counters = counter.Counter.all_from_dir(sys.argv))
    # test_outlier_removal(counters)
    # cumul_vs_panchenko(counters)

    # if by hand: change to the right directory before importing
    # import os; os.chdir(os.path.join(os.path.expanduser('~') , 'da', 'git', 'data'))
    # sys.argv = ['', 'disabled/bridge', '0.15.3-retrofixed/bridge/30.js', '0.15.3-retrofixed/bridge/70.js', '0.15.3-retrofixed/bridge/50.js']
    # sys.argv = ['', 'disabled/bridge', 'simple1/50', 'simple2/30', 'simple2/30-burst', 'simple1/10', 'simple2/5', 'simple2/20']
    # sys.argv = ['', 'disabled/bridge', 'wfpad/bridge', 'tamaraw']
    # sys.argv = ['', 'disabled/bridge', '0.22/10aI', '0.22/5aI', '0.22/5aII', '0.22/2aI__2016-07-23']
    # sys.argv = ['', 'disabled/bridge', 'wfpad/bridge', 'simple2/5', '0.22/5aI'] # TOP
    # PANCHENKO_PATH = os.path.join('..', 'sw', 'p', 'foreground-data', 'output-tcp')
    # counters = counter.Counter.all_from_panchenko(PANCHENKO_PATH)
    cross_test(sys.argv, with_svm=True) #, cumul=False)

### classification results per class
## defense_per_class is generated via gen_class_stats_list
## write gen_class_stats_list to csv file, transpose via
# with open('/tmp/names.csv', 'w') as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=defense_per_class[0].keys())
#     writer.writeheader()
#     for d in defense_per_class:
#             writer.writerow(d)

# read = []
# with open('/tmp/names.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     for row in reader:
#             read.append(row)
# a = np.array(read)
# b = a.transpose()
# with open('/tmp/names.csv', 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(b[8])
#     for row in b[:8]:
#             writer.writerow(_format_row(row))
#     for row in b[9:]:
#             writer.writerow(_format_row(row))
