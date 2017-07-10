'''computes trace statistics'''
import numpy as np

import collections

Stats = collections.namedtuple('Stats', ['tpi_mean', 'tpi_std'])


def total_packets_in_stats(counter_dict):
    '''Returns: dict - (mean, std) of each counter's total_packets_in'''
    out = {}
    for (k, v) in counter_dict.iteritems():
        tpi_list = _tpi_per_list(v)
        out[k] = (np.mean(tpi_list), np.std(tpi_list, ddof=1))


def _tpi_per_list(counter_list):
    return [x.get_tpi() for x in counter_list]
