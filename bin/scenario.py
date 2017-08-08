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


def _size_increase(base, compare):
    '''@return how much bigger/smaller is =compare= than =base= (in %)'''
    diff = {}
    if base.keys() != compare.keys():
        keys = set(base.keys())
        keys = keys.intersection(compare.keys())
        logging.warn("keys are different, just used %d common keys: %s",
                     len(keys), keys)
    else:
        keys = base.keys()
    for k in keys:
        diff[k] = float(compare[k][0]) / base[k][0]
    return 100 * (np.mean(diff.values()) - 1)

def _size_increase_helper(two_scenarios):
    return _size_increase(two_scenarios[two_scenarios.keys()[0]],
                          two_scenarios[two_scenarios.keys()[1]])

def size_increase_from_argv(scenario_argv, remove_small=True):
    '''computes sizes increases from sys.argv-like list, argv[1] is
baseline'''
    scenarios = counter.for_scenarios(
        scenario_argv[1:], remove_small=remove_small)
    stats = {k: _bytes_mean_std(v) for (k, v) in scenarios.iteritems()}
    out = {}
    for i in scenario_argv[2:]:
        out[i] = _size_increase(stats[scenario_argv[1]], stats[i])
    return out

# todo: code duplication: total_packets_in_stats
def _bytes_mean_std(counter_dict):
    '''@return a dict of {domain1: (mean1,std1}, ... domainN: (meanN, stdN)}
    >>> _bytes_mean_std({'yahoo.com': [counter._test(3)]})
    {'yahoo.com': (1800.0, 0.0)}
    '''
    out = {}
    for (domain, counter_list) in counter_dict.iteritems():
        total = [i.get_total_in() for i in counter_list]
        out[domain] = (np.mean(total), np.std(total))
    return out

# unused
def site_sizes(stats):
    '''@return {'url1': [size0, ..., sizeN-1], ..., urlM: [...]}

    stats = {k: _bytes_mean_std(v) for (k,v) in scenarios.iteritems()}'''
    scenarios = stats.keys()
    scenarios.sort()
    out = {}
    for url in stats[scenarios[0]].keys():
        out[url] = []
        for scenario in scenarios:
            out[url].append(stats[scenario][url][0])
    return out


def size_test(argv, outlier_removal=True):
    '''1. collect traces
    2. create stats
    3. evaluate for each vs first'''
    scenarios = counter.for_scenarios(argv[1:], remove_small=outlier_removal)
    stats = {k: _bytes_mean_std(v) for (k, v) in scenarios.iteritems()}
    scenario0 = argv[1]
    for scenario in argv[2:]:
        print '{}: {}'.format(scenario, _size_increase(stats[scenario0],
                                                      stats[scenario]))



# todo: code duplication: _bytes_mean_std
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
