#! /usr/bin/env python

# extracts (panchenko's) features from pcap and analyses them

import json
import logging
import math
import os
import os.path
import numpy as np
from sklearn import svm, neighbors, cross_validation
import subprocess
from sys import argv

HOME_IP='134.76.96.47'
#LOGLEVEL=logging.DEBUG
LOGLEVEL=logging.INFO
#LOGLEVEL=logging.WARN
TIME_SEPARATOR='@'

def _append_features(keys, filename):
    '''appends features in trace file "filename" to keys, indexed by domain'''
    domain = os.path.basename(filename).split(TIME_SEPARATOR)[0]
    if not keys.has_key(domain):
        keys[domain] = []
    keys[domain].append(Counter.from_pcap(filename))

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


class Counter:
    def __init__(self):
        self.bytes_in = 0
        self.bytes_out = 0
        self.packets_in = 0
        self.packets_out = 0
        self.packets = []
        self.timing = []
        self.warned = False

    @classmethod
    def from_json(cls, jsonstring):
        '''creates Counter from self.to_json'''
        tmp = Counter()

        for key, value in json.loads(jsonstring).iteritems():
            setattr(tmp, key, value)
        return tmp
        
    @classmethod
    def from_pcap(cls, filename):
        '''creates Counter from pcap file'''
        tmp = Counter()
    
        tshark = subprocess.Popen(args=['tshark','-r' + filename],
                                  stdout=subprocess.PIPE);
        for line in iter(tshark.stdout.readline, ''):
            logging.debug(line)
            try:
                (x, tstamp, src, y, dst, proto, size, rest) = line.split(None, 7)
            except ValueError:
                tmp.warned = True
                logging.warn('file: %s had problems in line \n%s\n', filename, line)
                break
            else:
                if not 'Len=0' in rest and not proto == 'ARP':
                    tmp._extract_line(src, size, tstamp)
                logging.debug('from %s to %s: %s bytes', src, dst, size)
        return tmp

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

    def _extract_line(self, src, size, tstamp):
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
        if hasattr(self, "size_markers"):
            return

        self.size_markers = [size / 600 for size in _sum_stream(self.packets)]
        # html response marker, thus request must have been sent before
        if self.size_markers[1] < 0:
            self.html_marker = - self.size_markers[1]
        else:
            self.html_marker = - self.size_markers[2]
        self.total_transmitted_bytes_in = self.bytes_in / 10000
        self.total_transmitted_bytes_out = self.bytes_out / 10000
        self.occuring_packet_sizes = (len(set(self.packets)) / 2) * 2
        self.percentage = (100 * self.packets_in / (self.packets_in + self.packets_out) / 5) * 5
        self.number_in = self.packets_in / 15
        self.number_out = self.packets_out / 15

    def to_json(self):
        '''prints packet trace to json, for reimport'''
        return json.dumps(self.__dict__)

def get_counters(*argv):
    '''get called as main, either creates counters for the arguments, or
    all files in this directory and below'''
    import doctest
    doctest.testmod()

    logging.basicConfig(format='%(levelname)s:%(message)s', level=LOGLEVEL)

    counters = {}
    if len(argv) > 1:
        for f in argv[1:]:
            _append_features(counters, f)
    else:
        for (dirpath, dirnames, filenames) in os.walk(os.getcwd()):
            l = len(filenames)
            for i,f in enumerate(filenames):
                fullname = os.path.join(dirpath, f)
                if TIME_SEPARATOR in fullname: # file like google.com@1445350513
                    logging.info('processing %s (%d/%d)', fullname, i, l)
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

    # svm
    svc = svm.SVC(kernel='linear')
    print 'svm: {}'.format(cross_validation.cross_val_score(svc, X, y, cv=5))
    # knn
    knn = neighbors.KNeighborsClassifier()
    print 'knn: {}'.format(cross_validation.cross_val_score(knn, X, y, cv=5))
    # svm rbf panchenko
    svcp = svm.SVC(C=2**17, gamma=2**(-19))
    print 'p-s: {}'.format(cross_validation.cross_val_score(svcp, X, y, cv=5))
    # svm liblinear
    svl = svm.LinearSVC()
    print 'l-s: {}'.format(cross_validation.cross_val_score(svl, X, y, cv=5))
