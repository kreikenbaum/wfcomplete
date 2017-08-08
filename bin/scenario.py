#! /usr/bin/env python
'''scenario-level operations

- computes trace statistics
- scenario name/date
'''
import numpy as np

import collections
import datetime
import doctest
import os


#Stats = collections.namedtuple('Stats', ['tpi', 'tpi_mean', 'tpi_std'])


class Scenario(object):
    '''just meta-information object'''
    def __init__(self, name):
        '''
        >>> Scenario('disabled/2016-11-13').date
        datetime.date(2016, 11, 13)
        >>> str(Scenario('disabled/2016-11-13'))
        'disabled'
        >>> Scenario('disabled/2016-11-13').name
        'disabled'
        >>> Scenario('disabled/05-12@10').date
        datetime.date(2016, 5, 12)
        >>> Scenario('disabled/05-12@10').size
        '10'
        >>> Scenario('disabled/bridge--2016-07-06').date
        datetime.date(2016, 7, 6)
        >>> hasattr(Scenario('disabled/bridge--2016-07-06'), "settting")
        False
        >>> Scenario('./0.22/10aI--2016-11-04-50-of-100').date
        datetime.date(2016, 11, 4)
        >>> Scenario('./0.22/10aI--2016-11-04-50-of-100').name
        '0.22'
        >>> Scenario('./0.22/10aI--2016-11-04-50-of-100').setting
        '10aI'
        >>> Scenario('wtf-pad/bridge--2016-07-05').date
        datetime.date(2016, 7, 5)
        >>> Scenario('wtf-pad/bridge--2016-07-05').name
        'wtf-pad'
        '''
        (self.name, date) = os.path.normpath(name).rsplit('/', 1)
        if '@' in date:
            (date, self.size) = date.split('@')
        date = date.replace('bridge--', '')
        if '--' in date:
            (self.setting, date) = date.split('--')
        # the following discards subset info: 10aI--2016-11-04-50-of-100
        tmp = [int(x) for x in date.split('-')[:3]]
        if len(tmp) == 2:
            tmp.insert(0, 2016)
        self.date = datetime.date(*tmp)


    def __str__(self):
        out = self.name
        # python 3.6: =if 'setting in self=
        if hasattr(self, 'setting'):
            out += ' with setting {}'.format(self.setting)
        return out


def total_packets_in_stats(counter_dict):
    '''Returns: dict - (mean, std) of each counter's total_packets_in'''
    out = {}
    for (k, v) in counter_dict.iteritems():
        tpi_list = _tpi_per_list(v)
        out[k] = (np.mean(tpi_list), np.std(tpi_list, ddof=1))


def tpi(counter_list):
    '''returns total incoming packets for each counter in list'''
    return [x.get_tpi() for x in counter_list]


doctest.testmod()
