#!/usr/bin/env python
'''aggregates trace data, extracts features'''
import doctest
import errno
import glob
import itertools
import json
import logging
import math
import os
import subprocess
import sys

import numpy as np

import pyshark
from pyshark.capture.capture import TSharkCrashException

DURATION_LIMIT_SECS = 8 * 60
# HOME_IP = '134.76.96.47' #td: get ips
HOME_IP = '134.169.109.25'
#LOGLEVEL = logging.DEBUG
LOGLEVEL = logging.INFO
LOGFORMAT = '%(levelname)s:%(filename)s:%(lineno)d:%(message)s'

MIN_CLASS_SIZE = 30
TOR_DATA_CELL_SIZE = 512

TIME_SEPARATOR = '@'

# defense->counter_dict cache-map
DEFENSES = {}

# module-globals
json_only = True
# td: remove this maybe (level==-1 did not increase accuracy vs 1)
minmax = None


def _append_features(keys, filename):
    '''appends features in trace file of name "filename" to keys.

    keys is a dictionary, indexed by domain, holding arrays of Counter'''
    domain = _extract_url(filename)
    if not keys.has_key(domain):
        keys[domain] = []
    counter = Counter.from_pcap(filename)
    if counter.packets and not counter.warned and counter.check():
        keys[domain].append(counter)
    else:
        logging.warn('%s discarded', counter.name)


def _discretize(number, step):
    '''discretizes(=round) number by increment, rounding up
    >>> _discretize(15, 3)
    15
    >>> _discretize(14, 3)
    15
    >>> _discretize(-3, 2)
    -2
    '''
    return int(math.ceil(float(number) / step) * step)

# td: rename to _get_url


def _extract_url(filename):
    '''extracts domain part from filename:
    >>> _extract_url('/tmp/google.com@12412')
    'google.com'
    '''
    return os.path.basename(filename).split(TIME_SEPARATOR)[0]


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


def _normalize(array):
    '''normalizes array so that its maximum absolute value is +- 1.0

    >>> _normalize([3,4,5])
    [0.6, 0.8, 1.0]
    >>> _normalize([-4, 2, -10])
    [-0.4, 0.2, -1.0]
    '''
    maxabs = float(max((abs(x) for x in array)))
    return [x / maxabs for x in array]


def _num_bytes(packets):
    '''number of (incoming, outgoing) bytes
    >>> _num_bytes([-3,1,-4,4])
    (5, 7)
    '''
    return((sum((x for x in packets if x > 0)),
            abs(sum((x for x in packets if x < 0)))))


def _num_packets(packets):
    '''number of (incoming, outgoing) packets
    >>> _num_packets([3,4,-1,-4, 1])
    (3, 2)
    '''
    return((sum((1 for x in packets if x > 0)),
            sum((1 for x in packets if x < 0))))


def _pad(row, upto=300):
    '''enlarges row to have upto entries (padded with 0)
    >>> _pad([2], 20)
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    '''
    row.extend([0] * (upto - len(row)))
    return row


def _remove_small_classes(counter_dict):
    '''@return dict with removed traces if there are less than
MIN_CLASS_SIZE per url'''
    out = {}
    for (k, v) in counter_dict.iteritems():
        if len(v) < MIN_CLASS_SIZE:
            logging.warn('class %s had only %s instances, removed',
                         k, len(v))
        else:
            out[k] = v
    return out


def _trace_append(X, y, y_names, x_add, y_add, name_add):
    '''appends single trace to X, y, y_names
    >>> X=[]; y=[]; y_n=[]; _trace_append(X,y,y_n,[1,2,3],0,'test'); X
    [[1, 2, 3]]
    >>> X=[]; y=[]; y_n=[]; _trace_append(X,y,y_n,[1,2,3],0,'test'); y_n
    ['test']
    >>> X=[]; y=[]; y_n=[]; _trace_append(X,y,y_n,[1,2,3],0,'test'); y
    [0]'''
    X.append(x_add)
    y.append(y_add)
    y_names.append(name_add)


def _trace_list_append(X, y, y_names, trace_list, method, list_id, name):
    '''appends list of traces to X, y, y_names'''
    for trace in trace_list:
        if not trace.warned:
            _trace_append(
                X, y, y_names, trace.__getattribute__(method)(), list_id, name)
        else:
            logging.warn('%s: one discarded', name)


def _test(num, val=600, millisecs=10.):
    '''@return Counter with num packets of size val each millisecs apart'''
    return from_json('{{"packets": {}, "timing": {}, "name": "test@0"}}'.format(
        [val] * num, _test_timing(num, val, millisecs)))


def _test_timing(num, val, millisecs):
    '''used for _test above'''
    out = []
    for i in range(num):
        out.append([i * millisecs / 1000., val])
    return out


def _sum_numbers(packets):
    '''sums number of adjacent packets in the same direction, discretized
    by 1,2,3-5,6-8,9-13,14

    >>> _sum_numbers([10, -1, -2, 4])
    [1, 2, 1]
    '''
    counts = _sum_stream([math.copysign(1, x) for x in packets])
    # extended as counts had 16, 21
    dictionary = {1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 4, 7: 4, 8: 4,
                  9: 5, 10: 5, 11: 5, 12: 5, 13: 5, 14: 6}
    return [dictionary[min(14, abs(x))] for x in counts]
    # return counts
    # return [math.log(abs(x), 2.4) for x in counts]


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

# td: separate json parsing and pcap-parsing
# td: use glob.iglob? instead of os.walk?


def all_from_dir(dirname, remove_small=True):
    '''All packets traces of the name domain@tstamp are parsed to
    Counters. If there are JSON files created by =save()=, use those
    instead of the pcap files for their domains. If there are neither,
    try to use a =batch= directory for Wang-style counters.
    '''
    global json_only
    out = {}

    filenames = glob.glob(os.path.join(dirname, "*"))
    for jfile in [x for x in filenames if '.json' in x]:
        domain = os.path.basename(jfile).replace('.json', '')
        logging.debug('traces for %s from JSON %s', domain, jfile)
        out[domain] = all_from_json(jfile)
        filenames.remove(jfile)
        for trace in [x for x in filenames if domain + '@' in x]:
            filenames.remove(trace)

    length = len(filenames)
    for i, filename in enumerate(filenames):
        if TIME_SEPARATOR in os.path.basename(filename):  # file like google.com@1445350513
            logging.info('processing (%d/%d) %s ', i + 1, length, filename)
            json_only = False
            _append_features(out, filename)
    if not out:
        out = all_from_wang(os.path.join(dirname, "batch"))
    if not out:
        out = all_from_panchenko(dirname)
    if not out:
        raise IOError('no counters in path "{}"'.format(dirname))
    if remove_small:
        return _remove_small_classes(out)
    else:
        return out


def all_from_json(filename):
    '''returns all the counters in json file named filename '''
    out = []
    for entry in open(filename):
        out.append(from_json(entry))
    return out


def all_from_panchenko(dirname='.'):
    '''@return a dict { class_1: [counters], ..., class_n: [c] }'''
    return dict(panchenko_generator(dirname))


def panchenko_generator(dirname):
    '''generator of Counters from filename, calling
from_panchenko_data()'''
    for filename in glob.glob(os.path.join(dirname, '*')):
        out = []
        with open(filename) as f:
            for line in f:
                out.append(Counter.from_panchenko_data(line))
        yield (os.path.basename(filename), out)


def all_from_wang(dirname="batch", cw=True):
    '''creates dict of Counters from wang's =batch/=-directory'''
    class_names = []
    try:
        with open("batch_list") as f:
            for line in f:
                (id, name) = line.split(':')
                assert int(id) is len(class_names)
                class_names.append(name.strip())
    except IOError:
        pass
    out = {}
    for filename in glob.glob(os.path.join(dirname, '*')):
        if filename[-1] is 'f':
            continue
        # td: need to deal with open world traces later
        if cw and '-' not in filename:
            continue
        (cls, inst) = os.path.basename(filename).split('-')
        if class_names:
            cls = class_names[int(cls)]
        if not cls in out:
            out[cls] = []
        with open(filename) as f:
            out[cls].append(Counter.from_wang(filename, cls, inst))
    return out


def dict_to_cai(counter_dict, writeto):
    '''write counter_dict's entries to writeto'''
    for counter_list in counter_dict.values():
        for c in counter_list:
            writeto.write('{}\n'.format(c.to_cai()))


def dict_to_panchenko(counter_dict, dirname='p_batch'):
    '''write counters in counter_dict to dirname/output-tcp/'''
    try:
        os.mkdir(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    try:
        os.mkdir(os.path.join(dirname, 'output-tcp'))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    for (k, v) in counter_dict.iteritems():
        with file(os.path.join(dirname, 'output-tcp', k), 'w') as f:
            list_to_panchenko(v, f)


def dict_to_wang(counter_dict):
    '''writes all counters in =counter_dict= to directory
    =./batch=. also writes a list number url to =./batch_list=
    '''
    os.mkdir("batch")
    batch_list = open("batch_list", "w")
    url_id = 0
    for (url, data) in counter_dict.iteritems():
        batch_list.write("{}: {}\n".format(url_id, url))
        for (counter_id, datum) in enumerate(data):
            datum.to_wang(os.path.join(
                "batch", "{}-{}".format(url_id, counter_id)))
        url_id += 1
    batch_list.close()


def _dirname_helper(dirname, clean=True):
    '''@return all traces in dirname'''
    previous_dir = os.getcwd()
    try:
        os.chdir(dirname)
        counter_dict = all_from_dir('.', remove_small=clean)
        if clean:
            counter_dict = outlier_removal(counter_dict)
    finally:
        os.chdir(previous_dir)
    return counter_dict


def dir_to_cai(dirname, clean=True):
    '''write all traces in defense in dirname to cai file'''
    counter_dict = _dirname_helper(dirname, clean)
    if dirname is '.':
        dirname = os.getcwd()
    with open(dirname.replace(os.sep, '___') + '.cai', 'w') as f:
        dict_to_cai(counter_dict, f)


def dir_to_herrmann(dirname, clean=True):
    '''write all traces in defense in dirname to herrmann libsvm file'''
    counter_dict = _dirname_helper(dirname, clean)
    # 2. write to file
    X, y, yd = to_features_herrmann(counter_dict)
    if dirname is '.':
        dirname = os.getcwd()
    to_libsvm(X, y, fname='her_svm')


def dir_to_wang(dirname, remove_small=True, outlier_removal_lvl=0):
    '''creates input to wang's classifier (in a =batch/= directory) for traces in dirname'''
    previous_dir = os.getcwd()
    os.chdir(dirname)
    counter_dict = all_from_dir('.', remove_small)
    if outlier_removal_lvl:
        counter_dict = outlier_removal(counter_dict, outlier_removal_lvl)
    dict_to_wang(counter_dict)
    os.chdir(previous_dir)


def dir_to_panchenko(dirname):
    '''creates input to CUMUL (in a =p-batch/= directory) for traces in
dirname'''
    counter_dict = _dirname_helper(dirname, clean=False)
    dict_to_panchenko(counter_dict)

# td: read up on ordereddict, maybe replace


def for_defenses(defenses, remove_small=True):
    '''@return dict: {defense1: {domain1: [counter1_1, ..., counter1_N],
    ..., domainN: [counterN_1, ... counterN_N]}, ..., defenseM:
    {domain1: [counter1_1, ..., counter1_N], ..., domainN:
    [counterN_1, ..., countersN_N]} for directories} in {@code
    defenses}
    '''
    out = {}
    if len(defenses) == 0:
        out['.'] = all_from_dir('.', remove_small=remove_small)
    for d in defenses:
        if d not in DEFENSES:
            DEFENSES[d] = all_from_dir(d, remove_small=remove_small)
        else:
            logging.info('reused defense: %s', d)  # debug?
        out[d] = DEFENSES[d]
    return out


def from_json(jsonstring):
    '''creates Counter from self.to_json-output'''
    tmp = Counter()

    for key, value in json.loads(jsonstring).iteritems():
        setattr(tmp, key, value)
    return tmp


def list_to_panchenko(counter_list, outfile):
    '''write counters to outfile, one per line'''
    for c in counter_list:
        outfile.write(c.to_panchenko() + '\n')


def save(counter_dict, prefix=''):
    '''saves counters as json data to file, each is prefixed with prefix'''
    for k, per_domain in counter_dict.iteritems():
        if not per_domain:
            continue
        with open(prefix + k + '.json', 'w') as file_:
            logging.debug('writing json for %s', k)
            for counter in per_domain:
                file_.write(counter.to_json())
                file_.write('\n')


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
                _trace_append(X_in, out_y, domain_names,
                              count.panchenko(max_lengths), class_number, domain)
            else:
                logging.warn('%s: one discarded', domain)
        class_number += 1
    return (np.array(X_in), np.array(out_y), domain_names)


def to_features_cumul(counter_dict):
    '''transforms counter data to CUMUL-feature vector pair (X,y)'''
    X_in = []
    out_y = []
    class_number = 0
    domain_names = []
    for domain, dom_counters in counter_dict.iteritems():
        if domain == "background":
            _trace_list_append(X_in, out_y, domain_names,
                               dom_counters, "cumul", -1, "background")
        else:
            _trace_list_append(X_in, out_y, domain_names,
                               dom_counters, "cumul", class_number, domain)
            class_number += 1
    return (np.array(X_in), np.array(out_y), domain_names)


def to_features_herrmann(counter_dict):
    '''@return herrmann et al's feature matrix from counters'''
    psizes = set()
    for counter_list in counter_dict.values():
        for counter in counter_list:
            psizes.update(counter.packets)
    list_psizes = list(psizes)
    # 3. for each counter: line: 'class, p1_occurs, ..., pn_occurs
    X_in = []
    class_number = 0
    domain_names = []
    out_y = []
    for domain, dom_counters in counter_dict.iteritems():
        for count in dom_counters:
            if not count.warned:
                X_in.append(count.herrmann(list_psizes))
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
                f.write('{}:{} '.format(no + 1, val))
        f.write('\n')


class Counter(object):

    '''single trace file'''

    def __init__(self, name=None):
        self.fixed = None
        self.name = name
        self.variable = None
        self.packets = []
        self.timing = []  # list [(p1_timing_secs, p1_size), ...]
        self.warned = False

    def __eq__(self, other):
        (cls, start) = self.name.split('@')
        (o_cls, o_start) = other.name.split('@')
        return (cls == o_cls
                and float(start) == float(o_start)
                and self.packets == other.packets)

    def __str__(self):
        return 'counter (packet, time): {}'.format(self.timing)

    @staticmethod
    def from_(*args):
        '''helper method to handle empty argument'''
        logging.info('args: %s, length: %d', args, len(args))
        if len(args) == 2:
            if os.path.isdir(args[1]):
                os.chdir(args[1])
                out = all_from_dir('.')
            else:
                out = {}
                _append_features(out, args[1])
        elif len(args) > 2:
            out = {}
            for filename in args[1:]:
                _append_features(out, filename)
        else:
            out = all_from_dir('.')
        return out

    @staticmethod
    def from_panchenko_data(line):
        '''creates Counter from panchenko's test data
        >>> Counter.from_panchenko_data('c.com 1234 1235:300').get_total_both()
        300
        >>> Counter.from_panchenko_data('c.com 1234 1235:300').name
        'c.com@1.234'
        >>> Counter.from_panchenko_data('c.com 1234 1235:300').packets
        [300]
        >>> Counter.from_panchenko_data('c.com 1234 1235:300').to_panchenko()
        'c.com 1234 1235:300'
        >>> Counter.from_panchenko_data('c.com 1234 1235:300').timing
        [[0.001..., 300]]
        >>> Counter.from_panchenko_data('c.com_exception 1234 1235:300').timing
        [[0.001..., 300]]
        >>> Counter.from_panchenko_data('c.com_exception 1234 1235:300').name
        'c.com@1_exception'
        '''
        tmp = Counter()

        elements = line.split()
        start_secs = int(elements[1]) / 1000.
        if '_' in elements[0]:
            (its_name, err) = elements[0].split('_', 1)
            tmp.name = '{}@{}_{}'.format(its_name, int(start_secs), err)
        else:
            tmp.name = '{}@{}'.format(elements[0], start_secs)
        for element in elements[2:]:
            (time_millis, value) = element.split(':')
            tmp.packets.append(int(value))
            tmp.timing.append([float(time_millis) / 1000 - start_secs,
                               int(value)])
        return tmp

    @staticmethod
    def from_pcap(filename):
        '''Creates Counter from pcap file. Less efficient than above.'''
        tmp = Counter.from_pcap_old(filename)
        if not tmp.warned:
            return tmp
        else:
            print '.',

        tmp = Counter(filename)

        cap = pyshark.FileCapture(filename)
        try:
            start = cap[0].sniff_time
            for pkt in cap:
                if 'ip' not in pkt or int(pkt.ip.len) == '52':
                    logging.debug('discarded from %s packet %s', filename, pkt)
                else:
                    tmp.extract_line(pkt.ip.src, pkt.ip.len,
                                     (pkt.sniff_time - start).total_seconds())
        except (KeyError, TSharkCrashException, TypeError):
            tmp.warned = True
            return tmp
        cap.close()
        return tmp


    @staticmethod
    def from_pcap_old(filename):
        '''creates Counter from pcap file'''
        tmp = Counter(filename)

        tshark = subprocess.Popen(args=['tshark',
                                        '-Tfields',
                                        '-eip.src', '-eip.len',
                                        '-eframe.time_relative',
                                        '-r' + filename],
                                  stdout=subprocess.PIPE)
        for line in iter(tshark.stdout.readline, ''):
            try:
                (src, size, time) = line.split()
            except ValueError:
                tmp.warned = True
                logging.debug('file: %s had problems in line \n%s\n',
                              filename, line)
                tshark.kill()
                break
            else:
                if int(size) == 52:
                    logging.debug('discarded line %s from %s', line, file)
                    continue
                tmp.extract_line(src, int(size)+14, time)
        return tmp


    @staticmethod
    def from_wang(filename, its_url=None, its_time=None):
        '''creates Counter from wang file (in batch dir, named "url-inst")'''
        tmp = Counter(filename)
        if not its_time and not its_url:
            (its_url, its_time) = filename.split('-')
        tmp.name = '{}@{}'.format(its_url, its_time)

        with open(filename) as f:
            for line in f:
                (secs, negcount) = line.split('\t')
                if abs(int(negcount)) <= 1:  # cell level
                    negcount = int(negcount) * TOR_DATA_CELL_SIZE
                tmp.packets.append(-int(negcount))
                tmp.timing.append([float(secs), -int(negcount)])
        return tmp

    def _variable_lengths(self):
        '''does the computation of lengths, assumes that variable is filled'''
        out = {}
        for k, feature in self.variable.iteritems():
            out[k] = len(feature)
        return out

    def check(self):
        '''if wrong, set warned flag and @return =false=, else =true='''
        if min(self.packets) > 0 or max(self.packets) < 0:
            logging.warn("file: %s's packets go only in one direction\n",
                         self.name)
            self.warned = True
            return False
        else:
            return True

    def cumul(self, num_features=100):
        '''@return CUMUL feature vector: inCount, outCount, outSize, inSize++'''
        c_abs = []
        # cumulated packetsizes
        c_rel = []
        in_size = 0
        out_size = 0
        in_count = 0
        out_count = 0

        # copied &modified from panchenko's generate-feature.py
        for packetsize in self.packets:
            if packetsize > 0:
                in_size += packetsize
                in_count += 1
                if len(c_rel) == 0:
                    c_rel.append(packetsize)
                    c_abs.append(packetsize)
                else:
                    c_rel.append(c_rel[-1] + packetsize)
                    c_abs.append(c_abs[-1] + abs(packetsize))
            elif packetsize < 0:
                out_size += abs(packetsize)
                out_count += 1
                if len(c_rel) == 0:
                    c_rel.append(packetsize)
                    c_abs.append(abs(packetsize))
                else:
                    c_rel.append(c_rel[-1] + packetsize)
                    c_abs.append(c_abs[-1] + abs(packetsize))
            else:
                logging.warn('packetsize == 0 in cumul')

        features = [in_count, out_count, out_size, in_size]
        cumulFeatures = np.interp(np.linspace(c_abs[0], c_abs[-1],
                                              num_features + 1),
                                  c_abs,
                                  c_rel)
        # could be cumulFeatures[1:], but never change a running system
        for el in itertools.islice(cumulFeatures, 1, None):
            features.append(el)

        return features

    def get_duration(self):
        '''@return duration of this trace'''
        try:
            return self.timing[-1][0]
        except IndexError:
            # panchenko input data
            return DURATION_LIMIT_SECS

    def get_total_in(self):
        '''returns total incoming bytes'''
        return _num_bytes(self.packets)[0]

    def get_total_both(self):
        '''@returns sum of incoming and outgoing bytes transferred in this'''
        (incoming, outgoing) = _num_bytes(self.packets)
        return incoming + outgoing

    def get(self, feature_name):
        '''returns the (scalar) feature of feature_name'''
        self._postprocess()
        return self.fixed[feature_name]

    def herrmann(self, p_sizes):
        '''@return bit vector: for each size in p_sizes, if counter has packet of that size'''
        out = [0] * len(p_sizes)
        for index, size in enumerate(p_sizes):
            if size in self.packets:
                out[index] = 1
        return out

    def panchenko(self, pad_by=300, extra=True):
        '''returns panchenko feature vector

        (html marker, bytes in/out, packet sizes in/out, total size,
        percentage incoming, number in/out, size markers, td:nummark)

        pad_by determines how much to pad feature-length vectors
        if pad_by is an int, all args will be padded by this amount
        if it is a dictionary, the corresponding values in
        self.variable will be padded

        extra determines whether to include extra features (duration)
        or return only Panchenko's features

        '''
        self._postprocess()
        if extra:
            out = self.fixed.values()
        else:
            out = [self.fixed[key]
                   for key in self.fixed.keys()
                   if key != 'duration']
        if isinstance(pad_by, int):
            tmp = {}
            for key in self.variable.keys():
                tmp[key] = pad_by
            pad_by = tmp
        for k, feature in self.variable.iteritems():
            out += _pad(feature, pad_by[k])
        return out

    def extract_line(self, src, size, tstamp):
        '''aggregates stream values'''
        if abs(int(size)) < TOR_DATA_CELL_SIZE:
            return
        if src == HOME_IP:  # outgoing negative as of panchenko 3.1
            self.packets.append(- int(size))
            self.timing.append((float(tstamp), - int(size)))
        elif src:  # incoming positive
            self.packets.append(int(size))
            self.timing.append((float(tstamp), int(size)))

    def _postprocess(self):
        '''sums up etc collected features'''
        if self.name is None:
            self.name = self.packets

        if self.fixed is not None:
            return

        self.fixed = {}
        self.variable = {}

        self.variable['size_markers'] = [_discretize(size, 600) for
                                         size in _sum_stream(self.packets)]
        # normalize size markers to remove redundant incoming at start
        # as the first chunk needs to be the html request, the second
        # the marker
        if self.variable['size_markers'][0] > 0:
            del self.variable['size_markers'][0]

        # html response marker, (request must have been sent before)
        self.fixed['html_marker'] = self.variable['size_markers'][1]
        # size_incoming size_outgoing
        (self.fixed['total_in'], self.fixed['total_out']) = (
            [_discretize(x, 10000) for x in _num_bytes(self.packets)])
        # number markers
        self.variable['number_markers'] = _sum_numbers(self.packets)
        # occurring packet sizes in + out
        self.fixed['num_sizes_out'] = (_discretize(len(set(
            [x for x in self.packets if x > 0])), 2))
        self.fixed['num_sizes_in'] = (_discretize(len(set(
            [x for x in self.packets if x < 0])), 2))
        # helper
        (packets_in, packets_out) = _num_packets(self.packets)
        # percentage incoming packets
        self.fixed['percentage_in'] = (
            _discretize(100 * packets_in / (packets_in + packets_out), 5))
        # number incoming
        self.fixed['count_in'] = _discretize(packets_in, 15)
        # number outgoing
        self.fixed['count_out'] = _discretize(packets_out, 15)
        # duration
        self.fixed['duration'] = self.timing[-1][0]
        # variable lengths test
        for i, val in enumerate(self._variable_lengths().values()):
            self.fixed['length_variable_' + str(i)] = val
        # all packets as of "A Systematic ..." svm.py code
        # self.variable['all_packets'] = self.packets # grew too big 4 mem
        return self

    # old doctest, old format
        # >>> a = _test(1); a.name = 'tps@1'; a.to_cai()
        # 'tps 600'
        # >>> a = _test(2); a.name = 'tps@1'; a.to_cai()
        # 'tps 600 600'
        # >>> a = _test(1, 500); a.name = 'tps@1'; a.to_cai()
        # 'tps'
        # >>> a = _test(3, 800); a.name = 'tps@1'; a.to_cai()
        # 'tps 600 600 600'
        # >>> a = _test(3, 1000); a.name = 'tps@1'; a.to_cai()
        # 'tps 1200 1200 1200'
    def to_cai(self, name=None):
        '''@return this as line in cai file (class, packets_rounded)
        >>> a = _test(1); a.name = 'tps@1'; a.to_cai()
        'tps +'
        >>> a = _test(2); a.name = 'tps@1'; a.to_cai()
        'tps + +'
        >>> a = _test(1, 500); a.name = 'tps@1'; a.to_cai()
        'tps'
        >>> a = _test(3, 800); a.name = 'tps@1'; a.to_cai()
        'tps + + +'
        >>> a = _test(3, 1000); a.name = 'tps@1'; a.to_cai()
        'tps + + +'
        '''
        if not name:
            name = self.name.split('@')[0]
        out = name
        for p in self.packets:
            while abs(p) >= 512:
                out += ' {}'.format('+' if np.sign(p) == 1 else '-')
                p -= np.sign(p) * 600
        return out

    def to_json(self):
        '''prints packet trace to json, for reimport'''
        return json.dumps(self.__dict__)

    def to_panchenko(self):
        '''@return line for this counter in Panchenko et al's format

        >>> a = _test(1); a.name = 'tps@1'; a.to_panchenko()
        'tps 1000 1000:600'
        >>> a = _test(2); a.name = 'tps@1'; a.to_panchenko()
        'tps 1000 1000:600 1010:600'
        '''
        (out, rest_secs) = self.name.split('@')
        try:
            start_millis = int(float(rest_secs) * 1000)
        except ValueError:
            (tmp_secs, err) = rest_secs.split('_')
            start_millis = int(float(tmp_secs) * 1000)
            out += '_{}'.format(err.replace(' ', '_').replace('\n', ''))
        out += ' {}'.format(start_millis)
        for (time_secs, val) in self.timing:
            out += ' {}:{}'.format(int(1000 * time_secs + start_millis), val)
        return out

    def to_wang(self, filename):
        '''writes this counter to filename in Wang et al's format'''
        with open(filename, 'w') as f:
            for (time, size) in self.timing:
                f.write("{}\t{}\n".format(time, -size))

    def variable_lengths(self):
        '''lengths of variable-length features'''
        self._postprocess()
        return self._variable_lengths()


class MinMaxer(object):

    '''keeps min and max scores

    >>> a = MinMaxer(); a.setIf(4,13); a.setIf(3,7); a.max
    13
    '''

    def __init__(self):
        self.min = sys.maxint
        self.max = - sys.maxint - 1

    def setIf(self, minval, maxval):
        if maxval > self.max:
            self.max = maxval
        if minval < self.min:
            self.min = minval

# outlier removal


def p_or_tiny(counter_list):
    '''removes if len(packets) < 2 or total_in < 2*512
    >>> len(p_or_tiny([_test(1), _test(3)]))
    1
    >>> len(p_or_tiny([_test(2, val=-600), _test(3)]))
    1
    '''
    return [x for x in counter_list
            if len(x.packets) >= 2 and x.get_total_in() >= 2 * 512]


def p_or_median(counter_list):
    '''removes if total_in < 0.2 * median or > 1.8 * median'''
    med = np.median([counter.get_total_in() for counter in counter_list])
    global minmax
    if minmax is None:
        minmax = MinMaxer()
    minmax.setIf(0.2 * med, 1.8 * med)

    return [x for x in counter_list
            if x.get_total_in() >= 0.2 * med and x.get_total_in() <= 1.8 * med]


def p_or_quantiles(counter_list):
    '''remove if total_in < (q1-1.5 * (q3-q1))
    or total_in > (q3+1.5 * (q3-q1)
    >>> [x.get_total_in()/600 for x in p_or_quantiles(map(_test, [0, 2, 2, 2, 2, 2, 2, 4]))]
    [2, 2, 2, 2, 2, 2]
    '''
    counter_total_in = [counter.get_total_in() for counter in counter_list]
    q1 = np.percentile(counter_total_in, 25)
    q3 = np.percentile(counter_total_in, 75)

    out = []
    # td: remove -1-code
    global minmax
    if minmax is None:
        minmax = MinMaxer()
    minmax.setIf(q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))
    for counter in counter_list:
        if (counter.get_total_in() >= (q1 - 1.5 * (q3 - q1)) and
                counter.get_total_in() <= (q3 + 1.5 * (q3 - q1))):
            out.append(counter)
    return out


def p_or_test(counter_list):
    '''outlier removal if training values are known'''
    return [x for x in counter_list
            if x.get_total_in() >= minmax.min
            and x.get_total_in() <= minmax.max]


def p_or_toolong(counter_list):
    '''@return counter_list with counters shorter than 8 minutes.

    The capturing software seemingly did not remove the others, even
    though it should have.'''
    return [x for x in counter_list if x.get_duration() < DURATION_LIMIT_SECS]


def outlier_removal(counter_dict, level=2):
    '''apply outlier removal to input of form
    {'domain1': [counter, ...], ... 'domainN': [..]}

    levels from 1 to 3 use panchenko's levels, -1 uses previous global minmax'''
    out = {}
    for (k, v) in counter_dict.iteritems():
        try:
            out[k] = p_or_tiny(v[:])
            out[k] = p_or_toolong(out[k])
            if level == -1:
                out[k] = p_or_test(out[k])
            if level > 2:
                out[k] = p_or_median(out[k])
            if level > 1:
                out[k] = p_or_quantiles(out[k])
            if not out[k]:
                raise ValueError
            logging.debug('%15s: outlier_removal(%d) removed %d from %d',
                          k, level, (len(v) - len(out[k])), len(v))
        except (ValueError, IndexError):  # somewhere, list got to []
            logging.warn('%s discarded in outlier removal', k)
    return out

doctest.testmod(optionflags=doctest.ELLIPSIS)
if __name__ == "__main__":
    logging.basicConfig(format=LOGFORMAT, level=LOGLEVEL)

    try:
        COUNTERS = Counter.from_(*sys.argv)
    except OSError:
        print 'needs a directory with pcap files named domain@timestamp (google.com@1234234234) or json files name domain.json (google.com.json)'
        sys.exit(1)

    if not json_only:
        save(COUNTERS)
        print 'counters saved to DOMAIN.json files'
    # if non-interactive, print timing data
    # from ctypes import pythonapi
    # if not os.environ.get('PYTHONINSPECT') and not pythonapi.Py_InspectFlag > 0:
    #     for t in itertools.chain.from_iterable(COUNTERS.values()):
    # was [trace for domain in COUNTERS.values() for trace in domain]:
    #         print t.timing
