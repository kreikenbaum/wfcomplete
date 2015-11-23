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


def test(X, y, estimator):
    '''tests estimator with X, y, prints type and result'''
    print estimator
    print cross_validation.cross_val_score(estimator, X, y, cv=5, n_jobs=-1)

if __name__ == "__main__":
    doctest.testmod()
    logging.basicConfig(format='%(levelname)s:%(message)s', level=LOGLEVEL)

    # if by hand: change to the right directory before importing
    # os.chdir('/home/w00k/da/sw/data/json/part')
    (X, y, y_domains) = to_features(counter.Counter.from_(sys.argv))

    test(X, y, svm.SVC(kernel='linear'))
    test(X, y, neighbors.KNeighborsClassifier())
    test(X, y, svm.SVC(C=2**17, gamma=2**(-19)))
    test(X, y, svm.LinearSVC())
    # grid rbf e-10 to e0
    Cs = np.logspace(-20, -11, base=10, num=10)
    Gs = np.logspace(-20, -11, base=10, num=10)
    for c in Cs:
        for gamma in Gs:
            test(X, y, svm.SVC(C=c, gamma=gamma))
    # end grid rbf
    # grid rbf focuseder
    cstart, cstop = -25, -15
    Cs = np.logspace(cstart, cstop, base=10, num=(abs(cstart - cstop)+1))
    gstart, gstop = -14, -7
    Gs = np.logspace(gstart, gstop, base=10, num=(abs(gstart - gstop)+1))
    for c in Cs:
        for gamma in Gs:
            test(X, y, svm.SVC(C=c, gamma=gamma))
    # end focuseder
    
