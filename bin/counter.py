#!/usr/bin/env python
'''aggregates trace data, extracts features'''
import doctest
import glob
import itertools
import json
import logging
import math
import numpy
import os
import subprocess
import sys

#HOME_IP = '134.76.96.47' #td: get ips
HOME_IP = '134.169.109.25'
#LOGLEVEL = logging.DEBUG
LOGLEVEL = logging.INFO
LOGFORMAT='%(levelname)s:%(filename)s:%(lineno)d:%(message)s'

TIME_SEPARATOR = '@'

json_only = True

def _append_features(keys, filename):
    '''appends features in trace file of name "filename" to keys.

    keys is a dictionary, indexed by domain, holding arrays of Counter'''
    domain = _get_domain(filename)
    if not keys.has_key(domain):
        keys[domain] = []
    counter = Counter.from_pcap(filename)
    if counter.packets and not counter.warned:
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

#td: rename to _get_url
def _get_domain(filename):
    '''extracts domain part from filename:
    >>> _get_domain('/tmp/google.com@12412')
    'google.com'
    '''
    return os.path.basename(filename).split(TIME_SEPARATOR)[0]

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

# td: move this back to analyse
def _ptest(num, val=600):
    '''returns counter with num packets of size val set for testing'''
    return Counter.from_json('{{"packets": {}}}'.format([val]*num))

def _sum_numbers(packets):
    '''sums number of adjacent packets in the same direction, discretized
    by 1,2,3-5,6-8,9-13,14

    >>> _sum_numbers([10, -1, -2, 4])
    [1, 2, 1]
    '''
    counts = _sum_stream([math.copysign(1, x) for x in packets])
    # extended as counts had 16, 21
    dictionary = {1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 4, 7: 4, 8:4,
                  9:5, 10:5, 11:5, 12:5, 13:5, 14:6}
    return [dictionary[min(14, abs(x))] for x in counts]
    #return counts
    #return [math.log(abs(x), 2.4) for x in counts]

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

def pad(row, upto=300):
    '''enlarges row to have upto entries (padded with 0)
    >>> pad([2], 20)
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    '''
    row.extend([0] * (upto - len(row)))
    return row

class Counter(object):
    '''single trace file'''
    def __init__(self, name=None):
        self.fixed = None
        self.name = name
        self.variable = None
        self.packets = []
        self.timing = [] # gets big
        self.warned = False

    def __str__(self):
        return 'p: {}'.format(self.panchenko())

    @staticmethod
    def from_json(jsonstring):
        '''creates Counter from self.to_json-output'''
        tmp = Counter()

        for key, value in json.loads(jsonstring).iteritems():
            setattr(tmp, key, value)
        return tmp

    @staticmethod
    def from_panchenko_data(line):
        '''creates Counter from panchenko's test data
        >>> Counter.from_panchenko_data('c.com 1234 1235:300').get_total_both()
        300
        '''
        tmp = Counter()

        elements = line.split()
        Counter.name = elements[0]
        for element in elements[2:]:
            (time, value) = element.split(':')
            tmp.packets.append(int(value))
            # could also append timing, but not used
        return tmp

    @staticmethod
    def all_from_panchenko(dirname='.'):
        '''creates list of Counters from filename, calling from_panchenko_data()'''
        out = {}
        os.chdir(dirname)
        for filename in glob.glob('*'):
            with open(filename) as f:
                out[filename] = []
                for line in f:
                    out[filename].append(Counter.from_panchenko_data(line))
        return out

    @staticmethod
    def save(counter_dict):
        '''saves counters as json data to file'''
        for k, per_domain in counter_dict.iteritems():
            if not per_domain:
                continue
            with open(k + '.json', 'w') as file_:
                logging.debug('writing json for %s', k)
                for counter in per_domain:
                    file_.write(counter.to_json())
                    file_.write('\n')

    @staticmethod
    def all_from_json(filename):
        '''returns all the counters in json file named filename '''
        out = []
        for entry in open(filename):
            out.append(Counter.from_json(entry))
        return out

    # td: separate json parsing and pcap-parsing
    # td: use glob.iglob? instead of os.walk?
    @staticmethod
    def all_from_dir(dirname):
        '''all packets traces in subdirectories of the name domain@tstamp are
        parsed to Counters. If there are JSON files created by
        =save()=, use those instead of the pcap files for their
        domains
        '''
        global json_only
        out = {}

        for (dirpath, _, filenames) in os.walk(dirname):
            for jfile in [x for x in filenames if '.json' in x]:
                domain = jfile.replace('.json', '')
                logging.info('traces for %s from JSON %s', domain, jfile)
                out[domain] = Counter.all_from_json(os.path.join(dirname,
                                                                 jfile))
                filenames.remove(jfile)
                for trace in [x for x in filenames if domain + '@' in x]:
                    filenames.remove(trace)

            length = len(filenames)
            for i, filename in enumerate(filenames):
                fullname = os.path.join(dirpath, filename)
                if TIME_SEPARATOR in fullname: # file like google.com@1445350513
                    logging.info('processing (%d/%d) %s ',
                                 i+1, length, fullname)
                    json_only = False
                    _append_features(out, fullname)
        if not out:
            logging.warn('%s did not contain any counter files', dirname)
        return out

    @staticmethod
    def from_(*args):
        '''helper method to handle empty argument'''
        logging.info('args: %s, length: %d', args, len(args))
        if len(args) == 2:
            try:
                os.chdir(args[1])
                out = Counter.all_from_dir('.')
            except OSError:
                pass # ok, was a filename
        elif len(args) > 1:
            out = {}
            for filename in args[1:]:
                _append_features(out, filename)
        else:
            out = Counter.all_from_dir('.')
        return out

    @classmethod
    def from_pcap(cls, filename):
        '''creates Counter from pcap file'''
        tmp = Counter(filename)

        tshark = subprocess.Popen(args=['tshark', '-r' + filename],
                                  stdout=subprocess.PIPE)
        for line in iter(tshark.stdout.readline, ''):
            logging.debug(line)
            try:
                (_, time, src, _, dst, proto, size, rest) = line.split(None, 7)
            except ValueError:
                tmp.warned = True
                logging.warn('file: %s had problems in line \n%s\n',
                             filename, line)
                break
            else:
                if not 'Len=0' in rest and not proto == 'ARP':
                    tmp.extract_line(src, size, time)
                logging.debug('from %s to %s: %s bytes', src, dst, size)
        return tmp

    def variable_lengths(self):
        '''lengths of variable-length features'''
        self._postprocess()
        return self._variable_lengths()

    def _variable_lengths(self):
        '''does the computation of lengths, assumes that variable is filled'''
        out = {}
        for k, feature in self.variable.iteritems():
            out[k] = len(feature)
        return out

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

    def panchenko(self, pad_by=300, extra=True):
        '''returns panchenko feature vector

        (html marker, bytes in/out, packet sizes in/out, total size,
        percentage incoming, number in/out, size markers, td:nummark)

        pad_by determines how much to pad feature-length vectors
        if pad_by is an int, all args will be pad()-ed by this amount
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
            out += pad(feature, pad_by[k])
        return out

    def extract_line(self, src, size, tstamp):
        '''aggregates stream values'''
        if src == HOME_IP: # outgoing negative as of panchenko 3.1
            self.packets.append(- int(size))
            self.timing.append((float(tstamp), - int(size)))
        elif src: #incoming positive
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
            _discretize(100 * packets_in /(packets_in + packets_out), 5))
        # number incoming
        self.fixed['count_in'] = _discretize(packets_in, 15)
        # number outgoing
        self.fixed['count_out'] = _discretize(packets_out, 15)
        # duration
        self.fixed['duration'] = self.timing[-1][0]
        # variable lengths test
        for i, val in enumerate(self._variable_lengths().values()):
            self.fixed['length_variable_'+str(i)] = val
        # all packets as of "A Systematic ..." svm.py code
        #        self.variable['all_packets'] = self.packets # grew too big 4 mem
        return self

    def to_json(self):
        '''prints packet trace to json, for reimport'''
        return json.dumps(self.__dict__)

    def cumul(self, num_features=100):
        '''@return CUMUL feature vector: inCount, outCount, outSize, inSize++'''
        c_abs = []
        # cumulated packetsizes
        c_rel = [] #td: good idea to have same name as method?
        inSize = 0
        outSize = 0
        inCount = 0
        outCount = 0

        # copied &modified from panchenko's generate-feature.py
        for packetsize in self.packets:
            if packetsize > 0:
                inSize += packetsize
                inCount += 1
                if len(c_rel) == 0:
                    c_rel.append(packetsize)
                    c_abs.append(packetsize)
                else:
                    c_rel.append(c_rel[-1] + packetsize)
                    c_abs.append(c_abs[-1] + abs(packetsize))
            elif packetsize < 0:
                outSize += abs(packetsize)
                outCount += 1
                if len(c_rel) == 0:
                    c_rel.append(packetsize)
                    c_abs.append(abs(packetsize))
                else:
                    c_rel.append(c_rel[-1] + packetsize)
                    c_abs.append(c_abs[-1] + abs(packetsize))
            else:
                logging.warn('packetsize == 0 in cumul')

        features = [inCount, outCount, outSize, inSize]
        cumulFeatures = numpy.interp(numpy.linspace(c_abs[0],
                                                    c_abs[-1],
                                                    num_features+1),
                                     c_abs,
                                     c_rel)
        # could be cumulFeatures[1:], but never change a running system
        for el in itertools.islice(cumulFeatures, 1, None):
            features.append(el)

        return features

# td: label lines together with one label
def plot(counter_list, color="blue", gnuplotter=None, title=None):
    '''plots counter's cumul data in color. in gnuplotter is not None, do reuse.

    usage: g = plot(COUNTERS['soso.com']);
    g2 = plot(COUNTERS['msn.com'], color="orange", gnuplotter=g)

    @return gnuplotter for additional plots'''
    import Gnuplot
    if title:
        datas = [Gnuplot.Data(counter_list.pop(0).cumul()[4:], inline=1,
                              with_='lines lc rgb "'+color+'"',
                              title=title)]
    else:
        datas = []
    datas.extend([Gnuplot.Data(x.cumul()[4:], inline=1,
                               with_='lines lc rgb "'+color+'"')
                  for x in counter_list])
    if not gnuplotter:
        gnuplotter = Gnuplot.Gnuplot()
    for d in datas:
        gnuplotter.replot(d)
    return gnuplotter

if __name__ == "__main__":
    doctest.testmod()
    logging.basicConfig(format=LOGFORMAT, level=LOGLEVEL)

    try:
        COUNTERS = Counter.from_(*sys.argv)
    except OSError:
        print 'needs a directory with pcap files named domain@timestamp (google.com@1234234234) or json files name domain.json (google.com.json)'
        sys.exit(1)

    if not json_only:
        Counter.save(COUNTERS)
        print 'counters saved to DOMAIN.json files'
    ## if non-interactive, print timing data
    from ctypes import pythonapi
    if not os.environ.get('PYTHONINSPECT') and not pythonapi.Py_InspectFlag > 0:
        for t in itertools.chain.from_iterable(COUNTERS.values()):
         # was [trace for domain in COUNTERS.values() for trace in domain]:
            print t.timing
    
