#! /usr/bin/env python
'''Analyses (Panchenko's) features returned from Counter class'''

import numpy as np
from sklearn import cross_validation, ensemble, multiclass, neighbors, svm, tree
import doctest
import logging
import sys

import counter

JOBS_NUM = 3
#JOBS_NUM = 4 #maybe at duckstein, but for panchenko problematic
LOGLEVEL = logging.DEBUG
#LOGLEVEL = logging.INFO
#LOGLEVEL = logging.WARN
TIME_SEPARATOR = '@'

def average_bytes(counter_dict):
    '''@return the average size over all traces'''
    count = 0
    total = 0
    for c_domain in counter_dict.values():
        for counter in c_domain:
            count += 1
            total += counter.get_total_both()
    return float(total) / count

def average_duration(counter_dict):
    '''@return the average duration over all traces'''
    count = 0
    total = 0
    for c_domain in counter_dict.values():
        for counter in c_domain:
            count += 1
            total += counter.timing[-1][0]
    return float(total) / count

def find_max_lengths(counters):
    '''determines maximum lengths of variable-length features'''
    max_lengths = counters.values()[0][0].variable_lengths()
    all_lengths = []
    for domain, domain_values in counters.iteritems():
        logging.debug('domain %s to feature array', domain)
        for trace in domain_values:
            all_lengths.append(trace.variable_lengths())
    for lengths in all_lengths:
        for key in lengths.keys():
            if max_lengths[key] < lengths[key]:
                max_lengths[key] = lengths[key]
    return max_lengths

def max_dict(d1, d2):
    '''@return biggest elements in both dicts (deep max). They need to
have the same members (with possibly different values)
    >>> a = {3: 4, 4: 7}; b = {3: 5, 4: 6}; max_dict(a, b)
    {3: 5, 4: 7}
    '''
    out = {}
    for (k, v) in d1.iteritems():
        out[k] = max(v, d2[k])
    return out

def to_features(counters, max_lengths=None):
    '''transforms counter data to panchenko.v1-feature vector pair (X,y)

    if max_lengths is given, use that instead of trying to determine'''
    if not max_lengths:
        max_lengths = find_max_lengths(counters)

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

def to_X_y_dom_size_d(place, out_rm=True):
    '''@return (X,y,y_domains, avg_size, avg_duration)

    tuple for Counters in =place= after cumul (and outlier removal if
    out_rm == {@code True})
    '''
    cs = counter.Counter.all_from_dir(place)
    if out_rm:
        cs = panchenko_outlier_removal(cs)
    size = average_bytes(cs)
    duration = average_duration(cs)
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

# td: maybe test that enough instances remain...
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

def panchenko_outlier_removal(counters):
    '''apply outlier removal to input of form
    {'domain1': [counter, ...], ... 'domainN': [..]}'''
    out = {}
    for (k, v) in counters.iteritems():
        try:
            out[k] = p_or_quantiles(p_or_median(p_or_tiny(v)))
        except ValueError: ## somewhere, list got to []
            logging.warn('%s discarded in outlier removal', k)
    return out
        
### evaluation code
GOOD = [ensemble.ExtraTreesClassifier(n_estimators=250),
        ensemble.RandomForestClassifier(),
        neighbors.KNeighborsClassifier(),
        tree.DecisionTreeClassifier()]
ALL = GOOD[:]
ALL.extend([ensemble.AdaBoostClassifier(),
            svm.SVC(gamma=2**-4)])

def esti_name(estimator):
    '''@return name of estimator class'''
    return str(estimator.__class__).split('.')[-1].split("'")[0]

def _test(X, y, estimator=GOOD[0], nj=JOBS_NUM, verbose=True, scale=False):
    '''tests estimator with X, y, @return result (ndarray)'''
    if scale:
        X = np.copy(X)
        with np.errstate(divide='ignore', invalid='ignore'):
            X /= np.max(np.abs(X), axis=0)
        X = np.nan_to_num(X)
    return cross_validation.cross_val_score(estimator, X, y, cv=5, n_jobs=nj)

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

def my_grid(X, y,
#            cs=np.logspace(0, 10, 11, base=2),
            cs=[256, 512, 1024],
            gammas=np.logspace(3, 8, 6, base=2)):
    '''grid-search on fixed params'''
    best = None
    bestres = 0
    for c in cs:
        for g in gammas:
            clf = multiclass.OneVsRestClassifier(svm.SVC(gamma=g, C=c))
            res = _test(X, y, clf, scale=True)
            if not best or bestres.mean() < res.mean():
                best = clf
                bestres = res
            logging.debug('c: {:8} g: {:10} acc: {}'.format(c, g, res.mean()))
    print 'optimal svm: ' + str(best)
    if best.estimator.C in (cs[0], cs[-1]) or best.estimator.gamma in (gammas[0], gammas[-1]):
        logging.warn('optimal parameters found at the border. c:%f, g:%f',
                     best.estimator.C, best.estimator.gamma)
    return best

# def smart_grid(X, y):
#     '''gradient-descent grid-search'''
#     from scipy import optimize
#     fun = lambda (c,g): -_testcg(X, y, c, g)
#     optimize.minimize(fun, (1, 1), jac=False, 

def _testcg(X, y, c, gamma):
    '''cross-evaluates ovr.svc with parameters c,gamma on X, y'''
    clf = multiclass.OneVsRestClassifier(svm.SVC(C=c, gamma=gamma))
    return _test(X, y, clf, scale=True, nj=1).mean()

# def smart_grid(X, y, stepmin = 2**3,
#                c_low=float(2**0), c_high=float(2**40), 
#                g_low=float(2**-10), g_high=float(2**30)):
#     '''gradient-descent grid-search'''
#     # init
#     results = {}
#     c_step = (c_high - c_low) / 2
#     g_step = (g_high - g_low) / 2
#     for c in (c_low, c_low + c_step, c_high):
#         for g in (g_low, g_low + g_step, g_high):
#             results[(c, g)] = _testcg(X, y, c, g)
#     while c_step > stepmin and g_step > stepmin:
#         # find max value
#
# 

def outlier_removal_vs_without(counters):
    (X, y, y_domains) = to_features_cumul(panchenko_outlier_removal(counters))
    (X2, y2, y2_domains) = to_features_cumul(counters)
    _compare(X, y, X2, y2)

def cumul_vs_panchenko(counters):
    (X, y, y_domains) = to_features(counters)
    (X2, y2, y2_domains) = to_features_cumul(counters)
    _compare(X, y, X2, y2)

def _compare(X, y, X2, y2, estimators=GOOD):
    for esti in estimators:
        _test(X, y, esti)
        _test(X2, y2, esti)

def counter_get(place, outlier_removal=True):
    '''helper to get counters w/o outlier_removal'''
    if outlier_removal:
        return panchenko_outlier_removal(counter.Counter.all_from_dir('.'))
    else:
        return counter.Counter.all_from_dir('.')

def cross_test(argv, cumul=True, outlier_rm=True, with_svm=False):
    '''cross test on dir: 1st has training data, rest have test'''
    # call with 1: x-validate test that
    # call with 2+: train 1 (whole), test 2,3,4,...
    # generate
    # cs = counter.Counter.all_from_dir(place)
    # if out_rm:
    #     cs = panchenko_outlier_removal(cs)
    # size = average_bytes(cs)
    # duration = average_duration(cs)
    # return to_features_cumul(cs) + (size, duration)

    # if len(argv) < 2:
    #     c = counter_get('.', outlier_rm)
    # else:
    #     c = counter_get(argv[1], outlier_rm)

    # if len(argv) > 2:
    #     test = {}
    #     for place in argv[2:]:
    #         test[place] = counter_get(place, outlier_rm)

    # if cumul:
        
    

    if len(argv) < 2:
        (X, y, y_domains, size, d) = to_X_y_dom_size_d('.')
    else:
        (X, y, y_domains, size, d) = to_X_y_dom_size_d(argv[1])

    if len(argv) > 2:
        test = {}
        for place in argv[2:]:
            test[place] = to_X_y_dom_size_d(place)

    #evaluate
    if with_svm:
        svm = my_grid(X, y)
        GOOD.append(svm)

    print 'cross-validation on X,y'
    for esti in GOOD:
        scale = True if 'SVC' in str(esti) else False
        print esti_name(esti),
        res = _test(X, y, esti, scale=scale)
        print '{}, {}'.format(res.mean(), res)
    for (place, (X2, y2, _, size2, d2)) in test.iteritems():
        print '\ntrain on: {} VS test on: {} (overhead {}%)'.format(
            argv[1], place, 100.0*(size2/size -1))
        for esti in GOOD:
            scale = True if 'SVC' in str(esti) else False
            print '{}: {}'.format(esti_name(esti),
                                  _xtest(X, y, X2, y2, esti, scale=scale))


    #_test(X, y, svm.SVC(C=10**-20, gamma=4.175318936560409e-10))
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

if __name__ == "__main__":
    doctest.testmod()
    logging.basicConfig(format='%(levelname)s:%(message)s', level=LOGLEVEL)

    # counters = counter.Counter.all_from_dir(sys.argv))
    # test_outlier_removal(counters)
    # cumul_vs_panchenko(counters)

    # if by hand: change to the right directory before importing
    # import os
    # PATH = os.path.join(os.path.expanduser('~') , 'da', 'git', 'data')
    # os.chdir(PATH)
    # sys.argv = ['', 'disabled/06-09@10/', '0.18.2/json-10/a_i_noburst/', '0.18.2/json-10/a_ii_noburst/', '0.18.2/json-10/b_i_from_100', '0.15.3/json-10/cache', '0.15.3/json-10/nocache']
    # sys.argv = ['', 'disabled/wfpad', 'wfpad']
    cross_test(sys.argv, with_svm=True)

#    import os
#    PATH = os.path.join(os.path.expanduser('~') , 'da', 'git', 'sw', 'p',
#                        'foreground-data', 'output-tcp')
#    counters = counter.Counter.all_from_panchenko(PATH)
#    (X, y ,y_dom) = to_features_cumul(counters)
#_test(X, y, nj=1)
