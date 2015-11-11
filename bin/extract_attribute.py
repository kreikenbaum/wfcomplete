#! /usr/bin/env python
'''extracts (panchenko's) features from pcap and analyses them'''

import doctest
import logging
import numpy as np
from sklearn import svm, neighbors, cross_validation
import sys

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
    for lengths in [x.variable_lengths()
                    for dv in counters.values() for x in dv]:
        # faster would be to flatten
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
    # all to same length
    max_len = max([len(x) for x in X_in])
    for x in X_in:
        _pad(x, max_len)
    # traces into X, y
    return (np.array(X_in), np.array(out_y), domain_names)


if __name__ == "__main__":
    doctest.testmod()
    logging.basicConfig(format='%(levelname)s:%(message)s', level=LOGLEVEL)

    (X, y, domain_names) = to_features(counter.Counter.from_(sys.argv))

    # svm
    print 'svm'
    svc = svm.SVC(kernel='linear')
    print cross_validation.cross_val_score(svc, X, y, cv=5, n_jobs=-1)
    del svc # free space
    # knn
    print 'knn'
    knn = neighbors.KNeighborsClassifier()
    print cross_validation.cross_val_score(knn, X, y, cv=5, n_jobs=-1)
    del knn
    # svm rbf panchenko
    print 'panchenko-rbf-svm'
    svcp = svm.SVC(C=2**17, gamma=2**(-19))
    print cross_validation.cross_val_score(svcp, X, y, cv=5, n_jobs=-1)
    del svcp
    # svm liblinear
    print 'liblinear'
    svl = svm.LinearSVC()
    print cross_validation.cross_val_score(svl, X, y, cv=5, n_jobs=-1)
    del svl
    
