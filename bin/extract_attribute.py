#! /usr/bin/env python
'''extracts (panchenko's) features from pcap and analyses them'''

import doctest
import logging
import numpy as np
from sklearn import svm, neighbors, cross_validation
import sys

# if you import by hand, include the path for the counter-module via
# sys.path.append('/home/w00k/da/git/bin')
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


if __name__ == "__main__":
    doctest.testmod()
    logging.basicConfig(format='%(levelname)s:%(message)s', level=LOGLEVEL)

    # if by hand: change to the right directory before importing
    # os.chdir('/home/w00k/da/sw/data/json')
    (X, y, y_domains) = to_features(counter.Counter.from_(sys.argv))

    # svm
    print 'svc_linear'
    svc_linear = svm.SVC(kernel='linear')
    print cross_validation.cross_val_score(svc_linear, X, y, cv=5, n_jobs=-1)
    del svc_linear # free space
    # knn
    print 'knn'
    knn = neighbors.KNeighborsClassifier()
    print cross_validation.cross_val_score(knn, X, y, cv=5, n_jobs=-1)
    del knn
    # svm rbf panchenko
    print 'panchenko-rbf-svm'
    svc_rbf = svm.SVC(C=2**17, gamma=2**(-19))
    print cross_validation.cross_val_score(svc_rbf, X, y, cv=5, n_jobs=-1)
    del svc_rbf
    # svm liblinear
    print 'liblinear'
    svc_liblinear = svm.LinearSVC()
    print cross_validation.cross_val_score(svc_liblinear, X, y, cv=5, n_jobs=-1)
    del svc_liblinear
    # grid rbf e-4 to e0
    Cs = np.logspace(-4, 0, base=10, num=10)
    Gs = np.logspace(-4, 0, base=10, num=10)
    for c in Cs:
        for gamma in Gs:
              print '{}, {}'.format(c, gamma)
              svc_rbf = svm.SVC(C=c, gamma=gamma)
              print cross_validation.cross_val_score(svc_rbf, X, y, cv=5, n_jobs=-1)
    # end grid rbf
