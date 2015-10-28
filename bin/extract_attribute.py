import logging
import math
from sys import argv
import subprocess

HOME_IP='134.76.96.47'
#LOGLEVEL=logging.DEBUG
LOGLEVEL=logging.INFO
#LOGLEVEL=logging.WARN

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
    dictionary = {1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 4, 7: 4, 8:4, 9:5, 10:5, 11:5, 12:5, 13:5, 14:6}
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
        self.html_marker = None
        self.total_transmitted_bytes_in = None
        self.total_transmitted_bytes_out = None
        self.occuring_packet_sizes = None
        self.percentage = None
        self.number_in = None
        self.number_out = None

        def __str__(self):
        self.postprocess()
        table = '| {0:>15} | {1:>30} |\n'
        logging.info('packets_in: %d', self.packets_in)
        logging.info('packets_out: %d', self.packets_out)

        return (table.format('size_markers', self.size_markers)
                + table.format('html_marker', self.html_marker)
                + table.format('total_in', self.total_transmitted_bytes_in)
                + table.format('total_out', self.total_transmitted_bytes_out)
                + table.format('sizes', self.occuring_packet_sizes)
                + table.format('percentage', self.percentage)
                + table.format('number_in', self.number_in)
                + table.format('number_out', self.number_out))

        # return (table.format('', "outgoing", "incoming")
        #         + table.format('BYTES', self.bytes_out, self.bytes_in)
        #         + table.format('PACKETS', self.packets_out, self.packets_in)
        #         + 'time: {0}\n'.format(self.last_time)
        #         + str(_sum_stream(self.packets)) + '\n'
        #         + str(self.packets) +'\n'
        #         + str(self.timing) + '\n\n')


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

    def postprocess(self):
        if self.size_markers == None:
            self.size_markers = []
            for size in _sum_stream(self.packets):
                self.size_markers.append(size / 600)
        if self.html_marker == None:
            if self.size_markers[0] < 0:
                self.html_marker = self.size_markers[0]
            else:
                self.html_marker = size_markers[1]
        if self.total_transmitted_bytes_in == None:
            self.total_transmitted_bytes_in = - self.bytes_in / 10000
        if self.total_transmitted_bytes_out == None:
            self.total_transmitted_bytes_out = self.bytes_out / 10000
        if self.occuring_packet_sizes == None:
            self.occuring_packet_sizes = (len(set(self.packets)) / 2) * 2
        if self.percentage == None:
            self.percentage = (100 * self.packets_in / (self.packets_in + self.packets_out) / 5) * 5
        if self.number_in == None:
            self.number_in = self.packets_in / 15
        if self.number_out == None:
            self.number_out = self.packets_out / 15


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    logging.basicConfig(format='%(levelname)s:%(message)s', level=LOGLEVEL)
    counter = Counts()
    
    tshark = subprocess.Popen(args=['tshark','-r' + argv[1]],
                              stdout=subprocess.PIPE);
    for line in iter(tshark.stdout.readline, ''):
        (num, tstamp, src, x, dst, proto, size, rest) = line.split(None, 7)
        if not '[ACK]' in rest and not proto == 'ARP':
            counter._extract_values(src, size, tstamp)

            logging.debug('from %s to %s: %s bytes', src, dst, size)
    print counter
