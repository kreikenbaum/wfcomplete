#! /usr/bin/env python
'''extracts (panchenko's) features from pcap and analyses them'''

import doctest
import logging
import numpy as np
from sklearn import svm, neighbors, cross_validation
import sys

# if you import by hand, include the path for the counter-module via
# sys.path.append('/home/w00k/da/git/bin')
# sys.path.append('/home/mkreik/bin')
import counter

#LOGLEVEL = logging.DEBUG
LOGLEVEL = logging.INFO
#LOGLEVEL = logging.WARN
TIME_SEPARATOR = '@'


def _pad(row, upto):
    '''enlarges row to have upto entries (padded with 0)
    >>> _pad([2], 20)
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    '''
    row.extend([0] * (upto - len(row)))
    return row

def to_features(counters):
    '''transforms counter data to feature vector pair (X,y)'''
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
    feature = 0
    domain_names = []
    for domain, dom_counters in counters.iteritems():
        for count in dom_counters:
            if not count.warned:
                X_in.append(count.panchenko(max_lengths))
                out_y.append(feature)
                domain_names.append(domain)
            else:
                logging.warn('%s: one discarded', domain)
        feature += 1
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


def test(X, y, estimator):
    '''tests estimator with X, y, prints type and result'''
    print estimator
    result = cross_validation.cross_val_score(estimator, X, y, cv=5, n_jobs=-1)
    print '{}, mean = {}'.format(result, result.mean())


if __name__ == "__main__":
    doctest.testmod()
    logging.basicConfig(format='%(levelname)s:%(message)s', level=LOGLEVEL)

    # if by hand: change to the right directory before importing
    # os.chdir('/home/w00k/da/sw/data/json/part')
    # os.chdir('/mnt/data/2-top100dupremoved_cleaned/')
    (X, y, y_domains) = to_features(counter.Counter.from_(sys.argv))

    test(X, y, svm.SVC(kernel='linear'))
    test(X, y, neighbors.KNeighborsClassifier())
    test(X, y, svm.SVC(C=2**17, gamma=2**(-19)))
    test(X, y, svm.LinearSVC())
    #grid rbf
    cstart, cstop = -45, -35
    Cs = np.logspace(cstart, cstop, base=10, num=(abs(cstart - cstop)+1))
#    Gs = np.logspace(gstart, gstop, base=10, num=10*(abs(gstart - gstop)+1))
    gamma = 4.175318936560409e-10
    for c in Cs:
#        for gamma in Gs:
        test(X, y, svm.SVC(C=c, gamma=gamma))
    # random forest
    from sklearn import ensemble
    test(X, y, ensemble.RandomForestClassifier())
    # feature importance
    forest = ensemble.ExtraTreesClassifier(n_estimators=250)
    forest.fit(X, y)
    forest.feature_importances_
    # extra trees
    test(X, y, ensemble.ExtraTreesClassifier())
    # extratree param
    for num in range(50, 400, 50):
        test(X, y, ensemble.ExtraTreesClassifier(n_estimators=num))
    # decision tree
    from sklearn import tree
    test(X, y, tree.DecisionTreeClassifier())
    # adaboost
    test(X, y, ensemble.AdaBoostClassifier())
    # linear params
    cstart, cstop = -5, 5
    Cs = np.logspace(cstart, cstop, base=10, num=(abs(cstart - cstop)+1))
    for c in Cs:
        test(X, y, svm.SVC(C=c, kernel='linear'))
    # metrics (single)
    for dist in [distance.braycurtis, distance.canberra,
                 distance.chebyshev, distance.cityblock, distance.correlation,
                 distance.cosine, distance.euclidean, distance.sqeuclidean]:
        test(X, y, neighbors.KNeighborsClassifier(metric='pyfunc', func=dist))
    # td: knn + levenshtein
    from scipy.spatial import distance
    import math
    def mydist(x, y):
        fixedm = distance.sqeuclidean(x[:8], y[:8])
        xbounds1 = (10, 10+x[8])
        ybounds1 = (10, 10+y[8])
        variable1 = gdl.metric(x[xbounds1[0]:xbounds1[1]],
                               y[ybounds1[0]:ybounds1[1]])
        xbounds2 = (10+x[8], 10+x[8]+x[9])
        ybounds2 = (10+x[8], 10+x[8]+y[9])
        variable2 = gdl.metric(x[xbounds2[0]:xbounds2[1]],
                               y[ybounds2[0]:ybounds2[1]])
        return math.sqrt(fixedm + variable1 + variable2)
    
