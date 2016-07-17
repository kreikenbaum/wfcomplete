#! /usr/bin/env python
'''Analyses (Panchenko's) features returned from Counter class'''

import numpy as np
from sklearn import cross_validation, ensemble, multiclass, neighbors, svm, tree
import doctest
import logging
import sys

import counter

JOBS_NUM = 2 # 1 (version1 to 4-6 (cumul onduckstein)), maybe also -4, ...
# panchenko grid takes about 16% mem/cpu: 3-4 should be fine, 5 ok'ish, 6 limit
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

def _compare(X, y, X2, y2, estimators=GOOD):
    for esti in estimators:
        _test(X, y, esti)
        _test(X2, y2, esti)

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

# move to counter.py
def _gen_counters(places, outlier_removal=True):
    '''@return dict: {place1: {domain1: counters1_1, ...  domainN: countersN_1},
    ..., placeM: {domain1: counters1_M, ...  domainN: countersN_M}}
    for directories} in {@code places}'''
    out = {}
    if len(places) == 0:
        out['.'] = counter_get('.', outlier_removal)

    for p in places:
        out[p] = counter_get(p, outlier_removal)

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

def _test(X, y, estimator=GOOD[0], nj=JOBS_NUM, verbose=True, scale=False):
    '''tests estimator with X, y, @return result (ndarray)'''
    if scale:
        X = np.copy(X)
        with np.errstate(divide='ignore', invalid='ignore'):
            X /= np.max(np.abs(X), axis=0)
        X = np.nan_to_num(X)
    return cross_validation.cross_val_score(estimator, X, y, cv=5, n_jobs=nj)

def _testcg(X, y, c, gamma):
    '''cross-evaluates ovr.svc with parameters c,gamma on X, y'''
    clf = multiclass.OneVsRestClassifier(svm.SVC(C=c, gamma=gamma))
    return _test(X, y, clf, scale=True, nj=JOBS_NUM).mean()

def _xtest(X_train, y_train, X_test, y_test, estimator, scale=False):
    '''cross_tests with estimator'''
    if scale:
        X_train = np.copy(X_train)
        X_test = np.copy(X_test)
        with np.errstate(divide='ignore', invalid='ignore'):
            X_train /= np.max(np.abs(X_train), axis=0)
            X_test /= np.max(np.abs(X_test), axis=0)
        X_train = np.nan_to_num(X_train)
        X_test = np.nan_to_num(X_test)
    estimator.fit(X_train, y_train)
    return estimator.score(X_test, y_test)

# unused, but could be useful
def _times_mean_std(counter_dict):
    '''analyse timing data (time overhead)

    @return a dict of {domain1: (mean1,std1}, ... domainN: (meanN, stdN)}
    with mean and standard of timing data
    '''
    out = {}
    for (domain, counter_list) in counter_dict.iteritems():
        total = [counter.timing[-1][0] for counter in counter_list]
        out[domain] = (np.mean(total), np.std(total))
    return out

def compare_stats(dirs):
    '''@return a dict {dir1: {domain1: {...}, ..., domainN: {...}},
    dir2:..., ..., dirN: ...} with domain mean, standard distribution
    and labels
    '''
    places = _gen_counters(dirs)
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

# unused
def to_X_y_dom_size_d(place, out_rm=True):
    '''@return (X,y,y_domains, avg_size, avg_duration)

    tuple for Counters in =place= after cumul (and outlier removal if
    out_rm == {@code True})
    '''
    cs = counter.Counter.all_from_dir(place)
    if out_rm:
        cs = panchenko_outlier_removal(cs)
    size = _average_bytes(cs)
    duration = _average_duration(cs)
    return to_features_cumul(cs) + (size, duration)

### outlier removal
def p_or_tiny(counter_list):
    '''removes if len(packets) < 2 or total_in < 2*512
    >>> len(p_or_tiny([counter._ptest(1), counter._ptest(3)]))
    1
    >>> len(p_or_tiny([counter._ptest(2, val=-600), counter._ptest(3)]))
    1
    '''
    return [x for x in counter_list
            if len(x.packets) >= 2 and x.get_total_in() >= 2*512]

def p_or_median(counter_list):
    '''removes if total_in < 0.2 * median or > 1.8 * median'''
    med = np.median([counter.get_total_in() for counter in counter_list])
    return [x for x in counter_list
            if x.get_total_in() >= 0.2 * med and x.get_total_in() <= 1.8 * med]

def p_or_quantiles(counter_list):
    '''remove if total_in < (q1-1.5 * (q3-q1))
    or total_in > (q3+1.5 * (q3-q1)
    >>> [x.get_total_in()/600 for x in p_or_quantiles(map(counter._ptest, [0, 2, 2, 2, 2, 2, 2, 4]))]
    [2, 2, 2, 2, 2, 2]
    '''
    counter_total_in = [counter.get_total_in() for counter in counter_list]
    q1 = np.percentile(counter_total_in, 25)
    q3 = np.percentile(counter_total_in, 75)

    out = []
    for counter in counter_list:
        if (counter.get_total_in() >= (q1 - 1.5 * (q3 - q1)) and
            counter.get_total_in() <= (q3 + 1.5 * (q3 - q1))):
            out.append(counter)
    return out

# td: maybe test that enough instances remain...
def panchenko_outlier_removal(counters):
    '''apply outlier removal to input of form
    {'domain1': [counter, ...], ... 'domainN': [..]}'''
    out = {}
    for (k, v) in counters.iteritems():
        try:
#            out[k] = p_or_quantiles(p_or_median(p_or_tiny(v)))
            out[k] = p_or_quantiles(p_or_tiny(v))
        except ValueError: ## somewhere, list got to []
            logging.warn('%s discarded in outlier removal', k)
    return out
        

def esti_name(estimator):
    '''@return name of estimator class'''
    return str(estimator.__class__).split('.')[-1].split("'")[0]

def my_grid(X, y,
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
            clf = multiclass.OneVsRestClassifier(svm.SVC(gamma=g, C=c))
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
        return my_grid(X, y,
                       new_search_range(best.estimator.C),
                       new_search_range(best.estimator.gamma),
                       results)
    else:
        return best, results

def new_search_range(best_param):
    '''returns new array of parameters to search in

    (use if best result was at border)
    '''
    return [best_param / 2, best_param, best_param * 2]

def smart_grid(X, y, stepmin=2, c_low=-2, c_high=18, g_low=-12, g_high=8):
    '''''smarter'' grid-search, params are exponents of 2

    use this to get the general range of the correct parameters for
    my_grid, as accuracy was slightly worse than for my_grid()
    '''
    import math
    # init
    results = {}
    c_bound = (c_low, c_high)
    g_bound = (g_low, g_high)
    while c_bound[1] - c_bound[0] > stepmin and g_bound[1] - g_bound[0] > stepmin:
        print 'iteration with bounds: {}, {}'.format(c_bound, g_bound)
        for c in np.logspace(c_bound[0], c_bound[1], 5, base=2):
            for g in np.logspace(g_bound[0], g_bound[1], 5, base=2):
                if (c,g) not in results:
                    results[(c,g)] = _testcg(X, y, c, g)
        # find max, reset bounds, continue
        best = None
        maxval = 0
        for ((c,g), v) in results.iteritems():
            if v > maxval:
                best = (math.log(c), math.log(g))
                maxval = v
        if best[0] in c_bound:
            c_plus = 0.5* (c_bound[1] - c_bound[0])
        else:
            c_plus = 0.25* (c_bound[1] - c_bound[0])
        if best[1] in g_bound:
            g_plus = 0.5* (g_bound[1] - g_bound[0])
        else:
            g_plus = 0.25* (g_bound[1] - g_bound[0])
        c_bound = (best[0] - c_plus, best[0] + c_plus)
        g_bound = (best[1] - g_plus, best[1] + g_plus)
    return multiclass.OneVsRestClassifier(svm.SVC(C=2**best[0], gamma=2**best[1]))

def outlier_removal_vs_without(counters):
    (X, y, y_domains) = to_features_cumul(panchenko_outlier_removal(counters))
    (X2, y2, y2_domains) = to_features_cumul(counters)
    _compare(X, y, X2, y2)

def cumul_vs_panchenko(counters):
    '''tests version1 and cumul on counters'''
    (X, y, y_domains) = to_features(counters)
    (X2, y2, y2_domains) = to_features_cumul(counters)
    _compare(X, y, X2, y2)

# td: remove this, or happens later
# td: move to counter.py
def counter_get(place, outlier_removal=True):
    '''helper to get counters w/o outlier_removal'''
    if outlier_removal:
        return panchenko_outlier_removal(counter.Counter.all_from_dir(place))
    else:
        return counter.Counter.all_from_dir(place)

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

def verbose_test_11(X, y, estimator):
    '''cross-test (1) estimator on (1) X, y, print results and estimator name'''
    scale = True if 'SVC' in str(estimator) else False
    print esti_name(estimator),
    res = _test(X, y, estimator, scale=scale)
    print '{}, {}'.format(res.mean(), res)

def cross_test(argv, cumul=True, outlier_rm=True, with_svm=False):
    '''cross test on dirs: 1st has training data, rest have test

    argv is like sys.argv, cumul triggers CUMUL. if false: panchenko 1'''
    # call with 1: x-validate test that
    # call with 2+: train 1 (whole), test 2,3,4,...
    places = _gen_counters(argv[1:], False)
    stats = {k: _bytes_mean_std(v) for (k,v) in places.iteritems()}
    sizes = {k: _average_bytes(v) for (k,v) in stats.iteritems()}
    # td: continue here, recompute duration (was not averaged per domain), compare
    # durations = {k: _average_duration(v) for (k,v) in places.iteritems()}

    place0 = argv[1] if len(argv) > 1 else '.'
    # or for first element only
    places[place0] = panchenko_outlier_removal(places[place0])

    # X,y for training set
    (train, test) = tts(places[place0])
    if cumul:
        (X, y, _) = to_features_cumul(train)
    else:
        (X, y, _) = to_features(train)

    if with_svm:
        clf,_ = my_grid(X, y)
#        clf = smart_grid(X, y) #enabled if ranges are not clear
        logging.info('grid result: %s', clf)
        GOOD.append(clf)

    # X,y for eval
    if cumul:
        (X, y, _) = to_features_cumul(places[place0])
    else:
        (X, y, _) = to_features(places[place0])
    # evaluate accuracy on all of unaddoned
    print 'cross-validation on X,y'
    for esti in GOOD:
        verbose_test_11(X, y, esti)

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

        for esti in GOOD:
            scale = True if 'SVC' in str(esti) else False
            print '{}: {}'.format(esti_name(esti),
                                  _xtest(X, y, X2, y2, esti, scale=scale))

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

# places = _gen_counters(sys.argv[1:])
# some_30 = top_30(means)
# timing = {k: _average_duration(v) for (k,v) in places.iteritems()}

if __name__ == "__main__":
    doctest.testmod()
    logging.basicConfig(format='%(levelname)s:%(message)s', level=LOGLEVEL)

    # counters = counter.Counter.all_from_dir(sys.argv))
    # test_outlier_removal(counters)
    # cumul_vs_panchenko(counters)

    # if by hand: change to the right directory before importing
    # import os; os.chdir(os.path.join(os.path.expanduser('~') , 'da', 'git', 'data'))
    # sys.argv = ['', 'disabled/06-17@100/', '0.18.2/json-100/b_i_noburst']
    # sys.argv = ['', 'disabled/06-17@10_from', '20.0/0_ai', '20.0/0_bi', '20.0/20_ai', '20.0/20_bi', '20.0/40_ai', '20.0/40_bi', '20.0/0_aii', '20.0/0_bii', '20.0/20_aii', '20.0/20_bii', '20.0/40_aii', '20.0/40_bii']
    # sys.argv = ['', 'disabled/bridge', 'wfpad/bridge', '22.0/10aI', 'simple1/10', 'simple1/50', '0.15.3-retrofixed/bridge/30.js', '0.15.3-retrofixed/bridge/70.js', '0.15.3-retrofixed/bridge/50.js', 'simple2/30', 'simple2/30-burst', 'tamaraw']
    # PANCHENKO_PATH = os.path.join('..', 'sw', 'p', 'foreground-data', 'output-tcp'); os.chdir(PANCHENKO_PATH)
    # counters = counter.Counter.all_from_panchenko(PANCHENKO_PATH)
    cross_test(sys.argv, with_svm=True) #, cumul=False)

