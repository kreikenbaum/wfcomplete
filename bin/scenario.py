#! /usr/bin/env python
'''scenario-level operations

- computes trace statistics
- scenario name/date
- load traces
'''
from __future__ import print_function

import copy
from dateutil import parser
import datetime
import doctest
import glob
import logging
import os
import random

import numpy as np

import config
import counter


DIR = os.path.join(os.path.expanduser('~'), 'da', 'git', 'data')
if os.uname()[1] == 'duckstein':
    DIR = os.path.join('/mnt', 'large')
RENAME = {
    "0.22": "new defense",
    "defense-client": "LLaMA",
    "disabled": "no defense",
    "llama": "LLaMA",
    "wf-cover": "new defense"
}
# were renamed on disk, hack to rename
PATH_RENAME = {
    "05-12@10": "05-12-2016--10@40",
    'bridge--2016-11-04-100@50': "10aI--2016-11-04--100@50",
    "10aI--2016-11-04-50-of-100": "10aI--2016-11-04--100@50",
    "json-10-nocache": "nocache--2016-06-17--10@30",
    "nobridge--2017-01-19-aI-factor=10": "nobridge-aI-factor=10--2017-01-19",
    "bridge--2016-09-21-100": "bridge--2016-09-21--100",
    "bridge--2016-09-26-100": "bridge--2016-09-26--100",
    "bridge--2016-08-30-100": "bridge--2016-08-30--100",
    "json-10/a-i-noburst": "a-i-noburst--2016-06-02--10@30",
    "json-10/a-ii-noburst": "a-ii-noburst--2016-06-03--10@30",
    "json-100/b-i-noburst": "b-i-noburst--2016-06-04--100@40",
    "30-burst": "30-burst--2016-07-11",
    "/30": "/30--2016-07-10",
    "/20": "/20--2016-07-17",
    "nobridge--2017-01-19-aI-factor=10-with-errors": "nobridge-aI-factor=10-with-errors--2017-01-19",
    "5--2016-09-23-100": "5--2016-09-23--100"
}
PATH_SKIP = [
    "../sw/w/, WANG14, knndata.zip",
    '../sw/w/, RND-WWW, disabled/foreground-data-subset',
    'disabled/foreground-data'
]


class Scenario(object):
    '''meta-information object with optional loading of traces'''
    def __init__(self, name, trace_args=None, smart=False, skip=False):
        ''' (further example usage in test.py)
        >>> Scenario('disabled/2016-11-13').date
        datetime.date(2016, 11, 13)
        >>> Scenario('disabled/2016-11-13').name # same as str(Scenario...)
        'no defense'
        >>> Scenario('disabled/05-12@10').num_sites
        10
        >>> Scenario('./0.22/10aI--2016-11-04-50-of-100').setting
        '10aI'
        '''
        self.traces = None
        self.trace_args = trace_args or config.trace_args()
        self.path = os.path.normpath(name)
        if name in PATH_SKIP or skip:
            self.name = name
            return
        # import pdb; pdb.set_trace()
        self.path = _prepend_if_ends(self.path, 'with-errors')
        self.path = _prepend_if_ends(self.path, 'with-errs')
        self.path = _prepend_if_ends(self.path, 'with7777')
        self.path = _prepend_if_ends(self.path, 'failure-in-between')
        for (pre, post) in PATH_RENAME.iteritems():
            if self.path.endswith(pre):
                self.path = self.path.replace(pre, post)
        if smart and not self.valid():
            self.path = list_all(self.path)[0].path
        try:
            (self.name, date) = self.path.rsplit('/', 1)
        except ValueError:
            self.name = os.path.normpath(self.path)
            return
        #import pdb; pdb.set_trace()
        numstr = None
        if '--' in date:
            if date.rindex('--') != date.index('--'):
                (self.setting, date, numstr) = date.split('--')
            else:
                try:
                    parser.parse(date.split('--')[1])
                    self.setting, date = date.split('--')
                except ValueError:
                    date, numstr = date.split('--')
        if '-' in date and '@' in date:
            (date, numstr) = date.rsplit('-', 1)
        if numstr:
            if '@' in numstr:
                self._num_sites, self.num_instances = [
                    int(x) for x in numstr.split('@')]
            else:
                self._num_sites = int(numstr)
        # the following discards subset info: 10aI--2016-11-04-50-of-100
        try:
            self.date = parser.parse(date).date()
        except ValueError:
            assert not hasattr(self, 'setting')
            logging.warn('failed to parse date for %s', name)
            self.setting = date
        # try:
        #     tmp = [int(x) for x in date.split('-')[:3]]
        #     if len(tmp) == 2:
        #         tmp.insert(0, 2016)
        #     try:
        #         self.date = datetime.date(*tmp)
        #     except TypeError:
        #         self.setting = date
        # except ValueError:
        #     self.setting = date
        try:
            with open(os.path.join(DIR, self.path, 'status')) as f:
                self.status = f.read()
        except IOError:
            self.status = "null"
        for (pre, post) in RENAME.iteritems():
            self.name = self.name.replace(pre, post)

    def __contains__(self, item):
        return item in self.path

    def _compareattr(self, other, attr):
        '''@return true if other has attr iff self has attr and values same'''
        return (hasattr(self, attr) and hasattr(other, attr)
                and getattr(self, attr) == getattr(other, attr)
                or (not hasattr(self, attr) and not hasattr(other, attr)))

    def __eq__(self, other):
        return (self._compareattr(other, "name")
                and self._compareattr(other, "date")
                and self._compareattr(other, "num_sites"))

    def __len__(self):
        '''@return the total number of instances in this scenario'''
        return sum((len(x) for x in self.get_traces().values()))

    def __str__(self):
        out = self.name
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

    def size_increase(self, trace_dict=None):
        '''@return size overhead of this vs closest disabled scenario'''
        return self._increase(size_increase, trace_dict)

    def time_increase(self, trace_dict=None):
        '''@return time overhead of this vs closest disabled scenario'''
        return self._increase(time_increase, trace_dict)

    def valid(self):
        '''@return whether the path is at least a directory'''
        return os.path.isdir(os.path.join(DIR, self.path))

    def to_dict(self):
        '''@return __dict__-like with properties'''
        # also without private members (_) etc
        return {a: getattr(self, a) for a in dir(self) if (
            not a.startswith('_')
            and not a == 'trace_args'
            and not a == 'traces'
            and not callable(getattr(self, a)))}


def _prepend_if_ends(whole, part):
    '''if whole ends with part, prepend it (modulo "-")

    >>> _prepend_if_ends('0.22/nobridge--2017-01-19-aI-factor=10-with-errors', \
                         'with-errors')
    '0.22/with-errors-nobridge--2017-01-19-aI-factor=10'
    '''
    if whole and whole.endswith('-' + part):
        splits = whole.rsplit('/', 1)
        last = part + '-' + splits[1].replace('-' + part, '')
        whole = splits[0] + '/' + last
    return whole


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
                               and not '/dump' in x
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
    '''@return increase in incoming bytes from base to compare'''
    base = _mean_std(base_trace_dict, "total_bytes_in")
    compare = _mean_std(compare_trace_dict, "total_bytes_in")
    return _compute_increase(base, compare)


def time_increase(base_trace_dict, compare_trace_dict):
    '''@return increase in duration from base to compare'''
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
        logging.debug("difference for %15s: %f", k, diff[k])
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
                if status['addon']['enabled'][x] is True]
    if len(enableds) > 1:
        logging.error("more than 1 addon enabled: %s", enableds)
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
## parse older "json" status
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
