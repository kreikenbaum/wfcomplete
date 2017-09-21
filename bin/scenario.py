#! /usr/bin/env python
'''scenario-level operations

- computes trace statistics
- scenario name/date
- load traces
'''
import numpy as np

import collections
import copy
import datetime
import doctest
import logging
import os
import random

import config
import counter


DIR = os.path.join(os.path.expanduser('~'), 'da', 'git', 'data')
TRACE_ARGS = {"remove_small": config.REMOVE_SMALL, "or_level": config.OR_LEVEL}


class Scenario(object):
    '''meta-information object with optional loading of traces'''
    def __init__(self, name, trace_args=TRACE_ARGS, smart=False):
        '''
        >>> Scenario('disabled/2016-11-13').date
        datetime.date(2016, 11, 13)
        >>> Scenario('disabled/2016-11-13').name # same as str(Scenario...)
        'disabled'
        >>> Scenario('disabled/05-12@10').size
        '10'
        >>> Scenario('./0.22/10aI--2016-11-04-50-of-100').setting
        '10aI'
        '''
        self.trace_args = trace_args
        self.path = os.path.normpath(name)
        if smart and not self.valid():
            self.path = list_all(self.path)[0].path
        try:
            (self.name, date) = self.path.rsplit('/', 1)
        except ValueError:
            self.name = os.path.normpath(self.path)
            return
        if '@' in date:
            (date, self.size) = date.split('@')
        if '--' in date:
            (self.setting, date) = date.split('--')
        # the following discards subset info: 10aI--2016-11-04-50-of-100
        try:
            tmp = [int(x) for x in date.split('-')[:3]]
            if len(tmp) == 2:
                tmp.insert(0, 2016)
            try:
                self.date = datetime.date(*tmp)
            except TypeError:
                self.setting = date
        except ValueError:
            self.setting = date
        try:
            with open(os.path.join(DIR, self.path, 'status')) as f:
                self.status = f.read()
        except IOError:
            self.status = None


    def __eq__(self, other):
        return self.path == other


    def __str__(self):
        out = self.name
        # python 3.6: =if 'setting in self=
        if hasattr(self, 'setting'):
            out += ' with setting {}'.format(self.setting)
        return out


    def __repr__(self):
        return '<scenario.Scenario("{}")>'.format(self.path)

        
    def binarize(self, bg_label='background', fg_label='foreground'):
        '''@return scenario with bg_label as-is, others combined to fg_label'''
        out = copy.copy(self)
        traces = {}
        traces[bg_label] = self.get_traces()[bg_label]
        traces[fg_label] = []
        for (domain, its_traces) in self.get_traces().iteritems():
            if domain != bg_label:
                traces[fg_label].extend(its_traces)
        setattr(out, 'traces', traces)
        return out


    def date_from_trace(self):
        trace = self.get_traces().values()[0][0]
        return datetime.datetime.fromtimestamp(
            float(trace.name.split('@')[1])).date()


    def valid(self):
        '''@return whether the path is at least a directory'''
        return os.path.isdir(os.path.join(DIR, self.path))


    def get_features_cumul(self):
        '''@return traces converted to CUMUL's X, y, y_domains'''
        X = []
        out_y = []
        class_number = 0
        domain_names = []
        for domain, dom_counters in self.get_traces().iteritems():
            if domain == "background":
                _trace_list_append(X, out_y, domain_names,
                                   dom_counters, "cumul", -1, "background")
            else:
                _trace_list_append(X, out_y, domain_names,
                                   dom_counters, "cumul", class_number, domain)
                class_number += 1
        return (np.array(X), np.array(out_y), domain_names)

    def get_open_world(self):
        '''@return scenario with traces and added background traces'''
        bg = self._closest('background', include_bg=True)
        self.get_traces()
        out = copy.copy(self)
        out.traces['background'] = bg.get_traces()['background']
        return out

    def get_sample(self, size, random_seed=None):
        '''@return sample of traces, each domain has size size'''
        random.seed(random_seed)
        out = {}
        for (domain, trace_list) in self.get_traces().iteritems():
            out[domain] = random.sample(trace_list, size)
        return out

    def get_traces(self):
        '''@return dict {domain1: [trace1, ..., traceN], ..., domainM: [...]}'''
        if not hasattr(self, "traces"):
            self.traces = counter.all_from_dir(os.path.join(DIR, self.path),
                                               **self.trace_args)
        return self.traces


    def size_increase(self, trace_dict=None):
        closest_disabled = self._closest("disabled")
        if closest_disabled == self:
            return 0
        return size_increase(closest_disabled.get_traces(),
                             trace_dict or self.get_traces())


    def _closest(self, filter_, include_bg=False):
        '''@return closest scenario that matches filter'''
        filtered = list_all(extra_filter=filter_, include_bg=include_bg)
        return min(filtered, key=lambda x: abs(self.date - x.date))


def list_all(extra_filter=None, include_bg=False, path=DIR):
    '''lists all scenarios in =path=.'''
    out = []
    for (dirname, _, _) in os.walk(path):
        if extra_filter and not extra_filter in dirname:
            continue
        out.append(dirname)
    out[:] = [x.replace(path+'/', './') for x in out]
    return [Scenario(x) for x in _filter_all(
        out, include_bg=include_bg)]

def _filter_all(all_, include_bg):
    '''If extra_filter is not None, only load scenario names matching the filter
    include_bg does at it says'''
    out = filter(lambda x: (not '/batch' in x
                            and not '/broken' in x
                            and not '/or' in x
                            and not '/output' in x
                            and not '/ow' in x
                            and not '/path' in x
                            and not '/p_batch' in x
                            and not '/results' in x
                            and not 'subsets/' in x
                            and (include_bg
                                 or (not '/background' in x
                                     and not '/bg' in x))
    ), all_)
    out[:] = [x for (i, x) in enumerate(out[:-1]) if (
        x not in out[i+1]
        or x+'-with-errors' == out[i+1])] + [out[-1]]
    return out


def size_increase(base_trace_dict, compare_trace_dict):
    base = _bytes_mean_std(base_trace_dict)
    compare = _bytes_mean_std(compare_trace_dict)
    return _size_increase_computation(base, compare)

def _size_increase_computation(base, compare):
    '''@return how much bigger/smaller =compare= is than =base= (in %)'''
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

# todo: code duplication: total_packets_in_stats
def _bytes_mean_std(trace_dict):
    '''@return a dict of {domain1: (mean1,std1}, ... domainN: (meanN, stdN)}
    >>> _bytes_mean_std({'yahoo.com': [counter._test(3)]})
    {'yahoo.com': (1800.0, 0.0)}
    '''
    out = {}
    for (domain, trace_list) in trace_dict.iteritems():
        total = [i.get_total_in() for i in trace_list]
        out[domain] = (np.mean(total), np.std(total))
    return out


def tpi(trace_list):
    '''returns total incoming packets for each trace in list'''
    return [x.get_tpi() for x in trace_list]
############# COPIED CODE, needed?
def _size_increase_helper(two_scenarios):
    return _size_increase_computation(two_scenarios[two_scenarios.keys()[0]],
                                      two_scenarios[two_scenarios.keys()[1]])

def size_increase_from_argv(scenario_argv, remove_small=True):
    '''computes sizes increases from sys.argv-like list, argv[1] is
baseline'''
    scenarios = counter.for_scenarios(
        scenario_argv[1:], remove_small=remove_small)
    stats = {k: _bytes_mean_std(v) for (k, v) in scenarios.iteritems()}
    out = {}
    for i in scenario_argv[2:]:
        out[i] = _size_increase_computation(stats[scenario_argv[1]], stats[i])
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
### END CODPIED CODE

# todo: code duplication: _bytes_mean_std
def total_packets_in_stats(trace_dict):
    '''Returns: dict - (mean, std) of each trace's total_packets_in'''
    out = {}
    for (k, v) in trace_dict.iteritems():
        tpi_list = _tpi_per_list(v)
        out[k] = (np.mean(tpi_list), np.std(tpi_list, ddof=1))


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


doctest.testmod()
