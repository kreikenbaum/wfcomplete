#! /usr/bin/env python

# extracts (panchenko's) features from pcap and analyses them

import logging
import math
import os
import os.path
import numpy as np
from sklearn import svm, neighbors
import subprocess
from sys import argv

HOME_IP='134.76.96.47'
#LOGLEVEL=logging.DEBUG
LOGLEVEL=logging.INFO
#LOGLEVEL=logging.WARN
TIME_SEPARATOR='@'

def _append_features(keys, fname):
    '''appends features in trace file "fname" to keys, indexed by domain'''
    domain = os.path.basename(fname).split(TIME_SEPARATOR)[0]
    if not keys.has_key(domain):
        keys[domain] = []
    keys[domain].append(analyze_file(fname))

def _enlarge(row, upto):
    '''enlarges row to have upto entries (padded with 0)
    >>> _enlarge([2], 20)
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    '''
    row.extend([0 for i in range(upto - len(row))])
    return row

def _sum_stream(packets):
    '''sums adjacent packets in the same direction
    >>> _sum_stream([10, -1, -2, 4])
    [10, -3, 4]
    '''
    tmp = 0
    out = []
    for size in packets:
        if size > 0 and tmp < 0 or size < 0 and tmp > 0:
            out.append(tmp)
            tmp = size
        else:
            tmp += size
    out.append(tmp)
    return out

def _sum_numbers(packets):
    '''sums number of adjacent packets in the same direction, grouped by
    1,2,3-5,6-8,9-13,14
    >>> _sum_numbers([10, -1, -2, 4])
    [1, 2, 1]
    '''
    counts = _sum_stream([math.copysign(1, x) for x in packets])
    dictionary = {1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 4, 7: 4, 8:4,
                  9:5, 10:5, 11:5, 12:5, 13:5, 14:6}
    return [dictionary[abs(x)] for x in counts]


class Counts:
    def __init__(self):
        self.bytes_in = 0
        self.bytes_out = 0
        self.packets_in = 0
        self.packets_out = 0
        self.last_time = ''
        self.packets = []
        self.timing = []
        self.size_markers = None
        self.html_marker = None # wrong, f.ex. for google.com@1445350513
        self.total_transmitted_bytes_in = None
        self.total_transmitted_bytes_out = None
        self.occuring_packet_sizes = None
        self.percentage = None
        self.number_in = None
        self.number_out = None
        self.warned = False

    def panchenko(self):
        '''returns panchenko feature vector, (html marker, bytes in, bytes
        out, total size, percentage incoming, packets in, packets out,
        size markers)'''
        self._postprocess()
        return ([self.html_marker,
                 self.total_transmitted_bytes_in,
                 self.total_transmitted_bytes_out,
                 self.occuring_packet_sizes,
                 self.percentage,
                 self.number_in,
                 self.number_out]
                + self.size_markers)

    def __str__(self):
        return 'p: {}'.format(self.panchenko())

    def _extract_values(self, src, size, tstamp):
        '''aggregates stream values'''
        if src == HOME_IP: # outgoing packet
            self.bytes_out += int(size)
            self.packets_out += 1
            self.packets.append(int(size))
            self.timing.append((tstamp, int(size)))
        else: #incoming
            self.bytes_in += int(size)
            self.packets_in += 1
            self.packets.append(- int(size))
            self.timing.append((tstamp, -int(size)))

        self.last_time = tstamp

    def _postprocess(self):
        '''sums up etc collected features'''
        if self.size_markers != None:
            return

        self.size_markers = [size / 600 for size in _sum_stream(self.packets)]
        if self.size_markers[0] < 0:
            self.html_marker = - self.size_markers[0]
        else:
            self.html_marker = - self.size_markers[1]
        self.total_transmitted_bytes_in = self.bytes_in / 10000
        self.total_transmitted_bytes_out = self.bytes_out / 10000
        self.occuring_packet_sizes = (len(set(self.packets)) / 2) * 2
        self.percentage = (100 * self.packets_in / (self.packets_in + self.packets_out) / 5) * 5
        self.number_in = self.packets_in / 15
        self.number_out = self.packets_out / 15

def analyze_file(filename):
    '''analyzes dump file, returns counter'''
    counter = Counts()
    
    tshark = subprocess.Popen(args=['tshark','-r' + filename],
                              stdout=subprocess.PIPE);
    for line in iter(tshark.stdout.readline, ''):
        logging.debug(line)
        try:
            (num, tstamp, src, x, dst, proto, size, rest) = line.split(None, 7)
        except ValueError:
            counter.warned = True
            logging.warn('file: %s had problems in line \n%s\n', filename, line)
            break
        else:
            if not '[ACK]' in rest and not proto == 'ARP':
                counter._extract_values(src, size, tstamp)
            logging.debug('from %s to %s: %s bytes', src, dst, size)

    return counter

def get_counters(*argv):
    '''get called as main, either prints out the arguments, or all files
    in this directory and below'''
    import doctest
    doctest.testmod()

    logging.basicConfig(format='%(levelname)s:%(message)s', level=LOGLEVEL)

    counters = {}
    if len(argv) > 1:
        for f in argv[1:]:
            _append_features(counters, f)
    else:
        for (dirpath, dirnames, filenames) in os.walk(os.getcwd()):
            for f in filenames:
                fullname = os.path.join(dirpath, f)
                if TIME_SEPARATOR in fullname: # file like google.com@1445350513
                    logging.info('processing %s', fullname)
                    _append_features(counters, fullname);

    return counters

def to_features(counters):
    '''transforms counter data to feature vector pair (X,y)'''
    X_in = []
    y = []
    feature = 0
    for domain, dom_counters in counters.iteritems():
        feature += 1
        for count in dom_counters:
            if not count.warned:
                X_in.append(count.panchenko())
                y.append(feature)
            else:
                logging.warn('%s: one discarded', domain)
    # all to same length
    max_len = max([len(x) for x in X_in])
    for x in X_in:
        _enlarge(x, max_len)
    # traces into X, y
    return (np.array(X_in), np.array(y))


if __name__ == "__main__":
    counters = get_counters(argv)
    (X, y) = to_features(counters)
    # shuffle
    np.random.seed(0)
    indices = np.random.permutation(len(y))
    percent = int(len(y) * 0.9)

    X_train = X[indices[:percent]]
    X_test = X[indices[percent:]]
    y_train = y[indices[:percent]]
    y_test = y[indices[percent:]]
    # svm
    svc = svm.SVC(kernel='linear')
    svc.fit(X_train, y_train)
    print 'svm score: {}'.format(svc.score(X_test, y_test))
    # knn
    knn = neighbors.KNeighborsClassifier()
    knn.fit(X_train, y_train)
    print 'knn score: {}'.format(knn.score(X_test, y_test))
    # svm rbf panchenko
    svcp = svm.SVC(C=2**17, gamma=2**(-19))
    svcp.fit(X_train, y_train)
    print 'svm(panchenko) score: {}'.format(svcp.score(X_test, y_test))
