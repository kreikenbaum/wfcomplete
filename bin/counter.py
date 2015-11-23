'''aggregates trace data, extracts features'''
import doctest
import json
import logging
import math
import os
import subprocess

HOME_IP = '134.76.96.47' #td: get ips
# LOGLEVEL = logging.DEBUG
LOGLEVEL = logging.INFO
TIME_SEPARATOR = '@'

def _append_features(keys, filename):
    '''appends features in trace file "filename" to keys

    keys is a dictionary, indexed by domain, holding arrays of Counter'''
    domain = _get_domain(filename)
    if not keys.has_key(domain):
        keys[domain] = []
    counter = Counter.from_pcap(filename)
    if not counter.warned:
        keys[domain].append(counter)
    else:
        logging.warn('%s discarded', counter.name)


def discretize(number, step):
    '''discretizes number by increment, rounding up
    >>> discretize(15, 3)
    15
    >>> discretize(14, 3)
    15
    '''
    return int(math.ceil(float(number) / step) * step)

def _get_domain(filename):
    '''extracts domain part from filename:
    >>> _get_domain('/tmp/google.com@12412')
    'google.com'
    '''
    return os.path.basename(filename).split(TIME_SEPARATOR)[0]

def pad(row, upto=300):
    '''enlarges row to have upto entries (padded with 0)
    >>> _pad([2], 20)
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    '''
    row.extend([0] * (upto - len(row)))
    return row

def sum_bytes(packets):
    '''sum of (incoming, outgoing) bytes
    >>> sum_bytes([-3,1,-4,4])
    (5, 7)
    '''
    return((sum((x for x in packets if x > 0)),
            sum((x for x in packets if x < 0))))

def sum_packets(packets):
    '''number of (incoming, outgoing) packets
    >>> sum_packets([3,4,-1,-4, 1])
    (3, 2)
    '''
    return((sum((1 for x in packets if x < 0)),
            sum((1 for x in packets if x > 0))))

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
    '''sums number of adjacent packets in the same direction, discretized
    by 1,2,3-5,6-8,9-13,14

    >>> _sum_numbers([10, -1, -2, 4])
    [1, 2, 1]
    '''
    counts = _sum_stream([math.copysign(1, x) for x in packets])
    # extended as counts had 16, 21
    dictionary = {1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 4, 7: 4, 8:4,
                  9:5, 10:5, 11:5, 12:5, 13:5, 14:6}
    #return [dictionary[min(14, abs(x))] for x in counts]
    #return counts
    print '2.4'
    return [math.log(abs(x), 2.4) for x in counts]

class Counter(object):
    '''single trace file'''
    def __init__(self, name=None):
        self.fixed = None
        self.name = name
        self.variable = None
        self.packets = []
        #        self.timing = [] # gets too big, else
        self.warned = False

    def __str__(self):
        return 'p: {}'.format(self.panchenko())

    @classmethod
    def from_json(cls, jsonstring):
        '''creates Counter from self.to_json-output'''
        tmp = Counter()

        for key, value in json.loads(jsonstring).iteritems():
            setattr(tmp, key, value)
        return tmp

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

    @staticmethod
    def all_from_dir(dirname):
        '''all packets traces in subdirectories of the name domain@tstamp are
        parsed to Counters. If there are JSON files created by
        save, use those instead of the pcap files for their domains'''
        out = {}

        for (dirpath, _, filenames) in os.walk(dirname):
            for jfile in [x for x in filenames if '.json' in x]:
                domain = jfile.replace('.json', '')
                logging.info('traces for %s from JSON %s', domain, jfile)
                out[domain] = Counter.all_from_json(jfile)
                filenames.remove(jfile)
                for trace in [x for x in filenames if domain + '@' in x]:
                    filenames.remove(trace)

            length = len(filenames)
            for i, filename in enumerate(filenames):
                fullname = os.path.join(dirpath, filename)
                if TIME_SEPARATOR in fullname: # file like google.com@1445350513
                    logging.info('processing (%d/%d) %s ',
                                 i+1, length, fullname)
                    _append_features(out, fullname)

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
        out = {}
        for k, feature in self.variable.iteritems():
            out[k] = len(feature)
        return out

    def panchenko(self, pad_by=300):
        '''returns panchenko feature vector

        (html marker, bytes in/out, packet sizes in/out, total size,
        percentage incoming, number in/out, size markers, td:nummark)

        pad_by determines how much to pad feature-length vectors
        if pad_by is an int, all args will be pad()-ed by this amount
        if it is a dictionary, the corresponding values in
        self.variable will be padded
        '''
        self._postprocess()
        out = self.fixed
        if type(pad_by) is int:
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
            #            self.timing.append((tstamp, - int(size)))
        elif src: #incoming positive
            self.packets.append(int(size))
            #            self.timing.append((tstamp, int(size)))

    def _postprocess(self):
        '''sums up etc collected features'''
        if self.name is None:
            self.name = self.packets
        logging.debug("_postprocess for %s", self.name)

        if self.fixed is not None:
            return

        self.fixed = []
        self.variable = {}

        self.variable['size_markers'] = [discretize(size, 600) for
                                         size in _sum_stream(self.packets)]
        # html response marker, (request must have been sent before)
        if self.variable['size_markers'][1] < 0:
            self.fixed.append(self.variable['size_markers'][1])
        else:
            self.fixed.append(self.variable['size_markers'][2])
        # size_incoming size_outgoing
        self.fixed.extend([discretize(x, 10000) for x in
                           sum_bytes(self.packets)])
        # number markers
        self.variable['number_markers'] = _sum_numbers(self.packets)
        # occurring packet sizes in + out
        self.fixed.append(discretize(len(set([x for x in
                                              self.packets if x > 0])), 2))
        self.fixed.append(discretize(len(set([x for x in
                                              self.packets if x < 0])), 2))
        # helper
        (packets_in, packets_out) = sum_packets(self.packets)
        # percentage incoming packets
        self.fixed.append(discretize(100 * packets_in
                                     /(packets_in + packets_out),
                                     5))
        # number incoming
        self.fixed.append(discretize(packets_in, 15))
        # number outgoing
        self.fixed.append(discretize(packets_out, 15))
        # all packets as of "A Systematic ..." svm.py code
#        self.variable['all_packets'] = self.packets # grew too big 4 mem

    def to_json(self):
        '''prints packet trace to json, for reimport'''
        return json.dumps(self.__dict__)

    @staticmethod
    def from_(*args):
        '''helper method to handle empty argument'''
        if len(args) > 1:
            out = {}
            for filename in args[1:]:
                _append_features(out, filename)
        else:
            out = Counter.all_from_dir('.')
        return out

if __name__ == "__main__":
    doctest.testmod()
    logging.basicConfig(format='%(levelname)s:%(message)s', level=LOGLEVEL)

    from sys import argv
    counters = Counter.from_(argv)
