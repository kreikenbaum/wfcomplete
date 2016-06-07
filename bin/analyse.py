#! /usr/bin/env python
'''Analyses (Panchenko's) features returned from Counter class'''

import numpy as np
from sklearn import svm, neighbors, cross_validation, ensemble, tree
import doctest
import logging
import sys

import counter

JOBS_NUM = 1
#JOBS_NUM = 4 #maybe at duckstein, but for panchenko problematic
#LOGLEVEL = logging.DEBUG
LOGLEVEL = logging.INFO
#LOGLEVEL = logging.WARN
TIME_SEPARATOR = '@'

def to_features(counters):
    '''transforms counter data to panchenko.v1-feature vector pair (X,y)'''
    max_lengths = counters.values()[0][0].variable_lengths()
    all_lengths = []
    for domain, domain_values in counters.iteritems():
        logging.info('domain %s to feature array', domain)
        for trace in domain_values:
            all_lengths.append(trace.variable_lengths())
#    for lengths in [x.variable_lengths()
#                    for dv in counters.values() for x in dv]:
        # faster would be to flatten
    for lengths in all_lengths:
        for key in lengths.keys():
            if max_lengths[key] < lengths[key]:
                max_lengths[key] = lengths[key]

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

def to_X_y_dom(place, outlier_removal=True):
    '''@return (X,y,y_domains) tuple for Counters in =place= after cumul
    and outlier removal if {@code True}'''
    cs = counter.Counter.all_from_dir(place)
    if outlier_removal:
        cs = panchenko_outlier_removal(cs)
    return to_features_cumul(cs)

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
        out[k] = p_or_quantiles(p_or_median(p_or_tiny(v)))
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

def _test(X, y, esti=GOOD[0], nj=JOBS_NUM, verbose=True):

    '''tests estimator with X, y, prints type and result'''
    result = cross_validation.cross_val_score(estimator, X, y, cv=5, n_jobs=nj)
    return result

def _xtest(X_train, y_train, X_test, y_test, esti):
    '''cross_tests with estimator'''
    esti.fit(X_train, y_train)
    return esti.score(X_test, y_test)

def my_grid(X, y, cs=np.logspace(11, 17, 7, base=2),
            gammas=np.logspace(-3, 3, 7, base=2)):
    '''grid-search on fixed params'''
    from sklearn.grid_search import GridSearchCV
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=0.25)
    svc = svm.SVC()
    clf = GridSearchCV(estimator=svc, verbose=2, n_jobs=-1,
                       param_grid=dict(C=cs, gamma=gammas))
    clf.fit(X_train, y_train)
    print 'best score: {}'.format(clf.best_score_)
    print 'with estimator: {}'.format(clf.best_estimator_)

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

def cross_test(argv):
    '''cross test on dir: 1st has training data, rest have test'''
    # call with 1: x-validate test that
    # call with 2+: train 1 (whole), test 2,3,4,...
    if len(argv) < 1:
        (X, y, y_domains) = to_X_y_dom('.', False)
    else:
        (X, y, y_domains) = to_X_y_dom(argv[1], True)

    if len(argv) > 1:
        test = {}
        for place in argv[1:]:
            test[place] = to_X_y_dom(place)

    print 'cross-validation'
    for esti in GOOD:
        print esti_name(esti),
        res = _test(X, y, esti)
        print '{}, {}'.format(res.mean(), res)
    for (place, (X2, y2, _)) in test.iteritems():
        print 'train on: {} VS test on: {}'.format(argv[0], place)
        for esti in GOOD:
            print '{}: {}'.format(esti_name(esti), _xtest(X, y, X2, y2, esti))

if __name__ == "__main__":
    doctest.testmod()
    logging.basicConfig(format='%(levelname)s:%(message)s', level=LOGLEVEL)

    # if by hand: change to the right directory before importing
    # PATH = os.path.join(os.path.expanduser('~') , 'da', 'git', 'data', 'json')
    # import os; os.chdir(os.path.join(PATH, 'disabled'))

    # counters = counter.Counter.all_from_dir(sys.argv))
    # test_outlier_removal(counters)
    # cumul_vs_panchenko(counters)

    # sys.argv = ['', 'disabled', 'a_i_noburst', 'a_ii_noburst']
    # cross_test(sys.argv)

    import os
    PATH = os.path.join(os.path.expanduser('~') , 'da', 'git', 'sw', 'p',
                        'foreground-data', 'output-tcp')
    counters = counter.Counter.all_from_panchenko(PATH)
    (X, y ,y_dom) = to_features_cumul(counters)
    _test(X, y, nj=1)
