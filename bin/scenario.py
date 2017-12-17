#! /usr/bin/env python
'''scenario-level operations

- computes trace statistics
- scenario name/date
- load traces
'''
from __future__ import print_function

import numpy as np

import collections
import copy
import datetime
import doctest
import glob
import logging
import os
import random

import config
import counter


DIR = os.path.join(os.path.expanduser('~'), 'da', 'git', 'data')
if os.uname()[1] == 'duckstein':
    DIR = os.path.join('/mnt', 'large')
RENAME = { "disabled": "no defense",
           "llama": "LLaMA",
           "defense-client": "LLaMA",
           "0.22": "new defense" }


class Scenario(object):
    '''meta-information object with optional loading of traces'''
    def __init__(self, name, trace_args=None, smart=False):
        '''
        >>> Scenario('disabled/2016-11-13').date
        datetime.date(2016, 11, 13)
        >>> Scenario('disabled/2016-11-13').name # same as str(Scenario...)
        'disabled'
        >>> Scenario('disabled/05-12@10').num_sites
        10
        >>> Scenario('./0.22/10aI--2016-11-04-50-of-100').setting
        '10aI'
        '''
        self.traces = None
        self.trace_args = trace_args or config.trace_args()
        self.path = os.path.normpath(name)
        if smart and not self.valid():
            self.path = list_all(self.path)[0].path
        try:
            (self.name, date) = self.path.rsplit('/', 1)
        except ValueError:
            self.name = os.path.normpath(self.path)
            return
        if '@' in date:
            (date, numstr) = date.split('@')
            try:
                self._num_sites = int(numstr)
            except ValueError:
                self._num_sites = int(numstr.split('-')[0])
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
            self.status = "null"


    def __contains__(self, item):
        return item in self.path


    def _compareattr(self, other, attr):
        '''@return true if other has attr iff self has attr and values same'''
        return (hasattr(self, attr) and hasattr(other, attr)
                and getattr(self, attr) == getattr(other, attr)
                or (not hasattr(self, attr) and not hasattr(other, attr)))

    def __eq__(self, other):
        return (self.name == other.name
                and self._compareattr(other, "date")
                and self._compareattr(other, "num_sites"))



    def __len__(self):
        '''@return the total number of instances in this scenario'''
        return sum((len(x) for x in self.get_traces().values()))


    def __str__(self):
        out = self.name
        for (pre, post) in RENAME.iteritems():
            out = out.replace(pre, post)
        # python 3.6: =if 'setting' in self=
        if hasattr(self, 'setting'):
            out += ' with setting {}'.format(self.setting)
        return out


    def __repr__(self):
        return '<scenario.Scenario("{}")>'.format(self.path)


    # idea: return whole list ordered by date-closeness
    def _closest(self, name_filter, include_bg=False, func_filter=None):
        '''@return closest scenario by date that matches filter, filtered by
size unless include_bg'''
        assert self.valid()
        filtered = list_all(name_filter, include_bg, func_filter=func_filter)
        if not include_bg:
            filtered = [x for x in filtered if x.num_sites == self.num_sites]
        return min(filtered, key=lambda x: abs(self.date - x.date))


    def _increase(self, method, trace_dict=None):
        try:
            closest_disabled = self._closest("disabled")
        except AttributeError:
            self.date = self.date_from_trace()
            closest_disabled = self._closest("disabled")
        if closest_disabled == self:
            return 0
        return method(closest_disabled.get_traces(),
                      trace_dict or self.get_traces())


    @property
    def num_sites(self):
        '''number of sites in this capture'''
        if hasattr(self, "_num_sites"):
            return self._num_sites
        else:
            return len(glob.glob(os.path.join(DIR, self.path, '*json')))


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
        '''retrieve date from traces'''
        self.trace_args = {'remove_small': False, 'or_level': 0}
        trace = self.get_traces().values()[0][0]
        return datetime.datetime.fromtimestamp(
            float(trace.name.split('@')[1])).date()


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


    def get_open_world(self, num="auto"):
        '''
        @return scenario with traces and (num) added background traces
        @param num: size of background set, if 'auto', use as many as fg set
        '''
        if 'background' in self.get_traces():
            logging.warn("scenario's traces already contain background set")
        background = self._closest('background', include_bg=True)
        self.get_traces()
        out = copy.copy(self)
        out.traces = copy.copy(self.traces)
        if num:
            if num == 'auto':
                num = sum([len(x) for x in self.traces.values()])
            out.traces['background'] = background.get_sample(num)['background']
        else:
            out.traces['background'] = background.get_traces()['background']
        return out


    def get_sample(self, num, random_seed=None):
        '''@return sample of traces, each domain has num traces'''
        random.seed(random_seed)
        out = {}
        for (domain, trace_list) in self.get_traces().iteritems():
            out[domain] = random.sample(trace_list, num)
        return out


    def get_traces(self):
        '''@return dict {domain1: [trace1, ..., traceN], ..., domainM: [...]}'''
        if not self.traces:
            self.traces = counter.all_from_dir(os.path.join(DIR, self.path),
                                               **self.trace_args)
        return self.traces


    def valid(self):
        '''@return whether the path is at least a directory'''
        return os.path.isdir(os.path.join(DIR, self.path))


    def size_increase(self, trace_dict=None):
        '''@return size overhead of this vs closest disabled scenario'''
        return self._increase(size_increase, trace_dict)


    def time_increase(self, trace_dict=None):
        '''@return time overhead of this vs closest disabled scenario'''
        return self._increase(time_increase, trace_dict)


def list_all(name_filter=None, include_bg=False, func_filter=None, path=DIR):
    '''lists all scenarios in =path=.'''
    out = []
    for (dirname, _, _) in os.walk(path):
        if dirname == path:
            continue
        if name_filter and not name_filter in dirname:
            continue
        out.append(dirname)
    out[:] = [x.replace(path+'/', './') for x in out]
    out = [Scenario(x) for x in _filter_all(
        out, include_bg=include_bg)]
    if func_filter:
        out = filter(func_filter, out)
    return out


def _filter_all(all_, include_bg):
    '''Filter out specific cases for scenario names,
    @param include_bg if True include background scenarios, else omit'''
    out = [x for x in all_ if (not '/batch' in x
                               and not '/broken' in x
                               and not '/foreground-data' in x
                               and not '/or' in x
                               and not '/output' in x
                               and not '/ow' in x
                               and not '/path' in x
                               and not '/p_batch' in x
                               and not '/results' in x
                               and not 'subsets/' in x
                               and (include_bg
                                    or (not '/background' in x
                                        and not '/bg' in x)))]
    out[:] = [x for (i, x) in enumerate(out[:-1]) if (
        x not in out[i+1]
        or x+'-with-errors' == out[i+1])] + [out[-1]]
    return out


def size_increase(base_trace_dict, compare_trace_dict):
    base = _mean_std(base_trace_dict, "total_bytes_in")
    compare = _mean_std(compare_trace_dict, "total_bytes_in")
    return _compute_increase(base, compare)


def time_increase(base_trace_dict, compare_trace_dict):
    base = _mean_std(base_trace_dict, "duration")
    compare = _mean_std(compare_trace_dict, "duration")
    return _compute_increase(base, compare)


def _compute_increase(base, compare):
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


def _mean_std(trace_dict, property_name):
    '''@return a dict of {domain1: (mean1,std1}, ... domainN: (meanN, stdN)}
    >>> _mean_std({'yahoo.com': [counter._test(3)]}, "total_bytes_in")
    {'yahoo.com': (1800.0, 0.0)}
    '''
    out = {}
    for (domain, trace_list) in trace_dict.iteritems():
        total = [getattr(i, property_name) for i in trace_list]
        out[domain] = (np.mean(total), np.std(total))
    return out


def tpi(trace_list):
    '''returns total incoming packets for each trace in list'''
    return [x.get_tpi() for x in trace_list]


def path_from_status(status, date=None):
    '''@return the scenario path from the status.sh output'''
    enableds = [x for x in status['addon']['enabled']
                if status['addon']['enabled'][x] == True]
    if len(enableds) > 1:
        logging.err("more than 1 addon enabled: %s", enableds)
    elif len(enableds) == 1:
        name = enableds[0].replace('@', '')
    else:
        name = "disabled"
    if not date:
        date = datetime.date.today()
    add = ''
    if name == "wf-cover":
        factor = status['addon']['factor']
        if not factor:
            factor = 50
        add = factor + 'aI--'
    if not date:
        date = datetime.date.today()
    return os.path.join(name, add + str(date))

############# COPIED CODE, needed?
def _size_increase_helper(two_scenarios):
    return _compute_increase(two_scenarios[two_scenarios.keys()[0]],
                             two_scenarios[two_scenarios.keys()[1]])

def size_increase_from_argv(scenario_argv, remove_small=True):
    '''computes sizes increases from sys.argv-like list, argv[1] is
baseline'''
    scenarios = counter.for_scenarios(
        scenario_argv[1:], remove_small=remove_small)
    stats = {k: _bytes_mean_std(v) for (k, v) in scenarios.iteritems()}
    out = {}
    for i in scenario_argv[2:]:
        out[i] = _compute_increase(stats[scenario_argv[1]], stats[i])
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
        print('{}: {}'.format(scenario, _size_increase(stats[scenario0],
                                                       stats[scenario])))
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


doctest.testmod(optionflags=doctest.ELLIPSIS)
# parse older "json" status
# json.loads(b.status.replace("'", '"').replace('False', 'false').replace('u"', '"'))

## scenarios without result
#a = {x: len(results.for_scenario(x)) for x in scenario.list_all()}
# filter(lambda x: x not in ',[]', str([x[0].path for x in filter(lambda x: x[1] == 0, a.iteritems())])) # unfiltered starts with ...[x[0

## weird scenario
# import mplot
# import results
# from sklearn import model_selection, preprocessing
# a = Scenario("disabled/bridge--2017-10-16")
# b = results.for_scenario(a)[0]
# X, y, domains = a.get_features_cumul()
# X = preprocessing.MinMaxScaler().fit_transform(X) # scaling is idempotent
# clf = b.get_classifier()
# y_pred = model_selection.cross_val_predict(clf, X, y, cv=config.FOLDS,
#                                                      n_jobs=config.JOBS_NUM)
# c = mplot.confusion(y, y_pred, domains, rotation=90)
