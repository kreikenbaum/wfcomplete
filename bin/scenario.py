#! /usr/bin/env python
'''scenario-level operations

- computes trace statistics
- scenario name/date
- load traces
'''
from __future__ import print_function

import copy
import datetime
import doctest
import glob
import itertools
import json
import logging
import os
import random
import re
import subprocess

import numpy as np

import config
import counter
import mymetrics
import sites
from capture import utils

NODEF = "defenseless"
# # muted
COLORS = {
    NODEF: "#d65f5f",            # 2
    "LLaMA": "#b47cc7",          # 3
    "LLaMA-nodelay": "#b47cc7",  # 3
    "SCT light": "#77bedb",      # 5
    "SCT medium": "#6acc65",     # 1
    "SCT heavy": "#4878cf",      # 0
    "SCT": "#4878cf"  # same as SCT heavy
}

INT_REGEXP = re.compile("-?([+-]?[0-9]+.*)")

NEW_DEFENSE = "SCT"

DIR = os.path.join(os.path.expanduser('~'), 'da', 'git', 'data')
if os.uname()[1] == config.OLD_HOST:
    DIR = os.path.join('/mnt', 'large')
RENAME = {
    "defense-client": "LLaMA",
    "defense-client-nodelay": "LLaMA-nodelay",
    "disabled": "defenseless",
    "external": "WANG14"
}
for _ in ["0.15.3", "0.18.2", "0.19", "0.20", "0.21", "0.22",
          "simple2", "simple", config.COVER_NAME]:
    RENAME[_] = NEW_DEFENSE

PATH_SKIP = [
    "../sw/w/, WANG14, knndata.zip",
    "../sw/w/, RND-WWW, disabled/foreground-data-subset",
    "disabled/foreground-data"
]


# code grew with scenario renames etc to keep db matching
class Scenario(object):
    '''meta-information about a scenario with optional loading of its traces'''
    def __init__(self, name, trace_args=None, smart=False, skip=False,
                 open_world=False, exclude_sites=[]):
        ''' (further example usage in test.py)
        >>> Scenario('disabled/2016-11-13').date
        datetime.date(2016, 11, 13)
        >>> Scenario('disabled/2016-11-13').name # same as str(Scenario...)
        'defenseless'
        >>> Scenario("disabled/2016-05-12--10@40").num_sites
        10
        >>> Scenario("0.22/10aI--2016-11-04--100@50").setting
        '10aI'
        '''
        self.traces = None
        self.trace_args = trace_args or config.trace_args()
        self._open_world_config = open_world
        self._exclude = exclude_sites
        self.path = os.path.normpath(name)
        if name in PATH_SKIP or skip or ".unison" in name:
            self.name = name
            self.date = datetime.date(1000, 1, 1)
            logging.debug("skipped " + name)
            return
        if smart and not self.valid():
            self.path = list_all(self.path)[0].path
        # todo: rename scenarios, also in db, remove this call
        path = _prepend_if_ends(
            self.path,
            'with-errors', 'with-errs', 'with7777', 'failure-in-between')
        try:
            (self.name, date) = path.rsplit('/', 1)
        except ValueError:
            logging.error("failed to parse path %s", path)
            return
        numstr = None
        if '--' in date:
            if date.rindex('--') != date.index('--'):
                (self.setting, date, numstr) = date.split('--')
            else:
                try:
                    datetime.datetime.strptime(date.split('--')[1],
                                               "%Y-%m-%d").date()
                    self.setting, date = date.split('--')
                except ValueError:
                    date, numstr = date.split('--')
        if numstr:
            if '@' in numstr:
                self._num_sites, self.num_instances = [
                    int(x) for x in numstr.split('@')]
            else:
                self._num_sites = int(numstr)
        self.date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
        try:
            with open(os.path.join(DIR, self.path, 'status')) as f:
                try:
                    self.status = json.load(f)
                except ValueError:
                    self.status = f.read()
        except IOError:
            self.status = "null"
        for (pre, post) in RENAME.iteritems():
            if pre == self.name:
                self.version = self.name
                self.name = self.name.replace(pre, post)
        if self.background:
            self.trace_args['or_level'] = 0

    def __lt__(self, other):
        return self.date < other.date

    def __contains__(self, item):
        return item in self.path

    def _compareattr(self, other, *attrs):
        '''@return true if other has attr iff self has attr and values same'''
        for attr in attrs:
            if not (hasattr(self, attr) and hasattr(other, attr)
                    and getattr(self, attr) == getattr(other, attr)
                    or (not hasattr(self, attr) and not hasattr(other, attr))):
                logging.debug("%r != %r on %s", self, other, attr)
                return False
        return True

    def __eq__(self, other):
        return self._compareattr(other, "name", "config", "date", "site")
#                and self._compareattr(other, "num_sites")) # failed # where?

    def __len__(self):
        '''@return the total number of instances in this scenario'''
        try:
            return sum((len(x) for x in self.traces.values()))
        except AttributeError:
            logging.info("guessed size")
            return self.num_instances * self.num_sites

    def __str__(self):
        return '{} on {}'.format(self.describe(), self.date)

    def __repr__(self):
        return '<scenario.Scenario("{}")>'.format(self.path)

    # idea: return whole list ordered by date-closeness
    def _closest(self, in_name=None, include_bg=False, filter_scenario=None):
        '''@return closest scenario by date that matches filter, filtered to
        have at least the scenario's number of sites, unless
        include_bg==True'''
        assert self.valid()
        filtered = list_all(
            in_name, include_bg, filter_scenario=filter_scenario)
        if not include_bg:
            filtered = [x for x in filtered if x.num_sites >= self.num_sites]
        return min(filtered, key=lambda x: abs(self.date - x.date))

    def _increase(self, method, trace_dict=None):
        try:
            closest_disabled = self._closest(
                "disabled",
                filter_scenario=lambda so: so.uses_bridge == self.uses_bridge)
        except AttributeError:
            self.date = self.date_from_trace()
            closest_disabled = self._closest(
                "disabled",
                filter_scenario=lambda so: so.uses_bridge == self.uses_bridge)
        if closest_disabled == self:
            return 0
        return method(closest_disabled.get_traces(current_sites=False),
                      trace_dict or self.get_traces(current_sites=False))

    def binarized(self, bg_label='background', fg_label='foreground'):
        '''@return scenario with bg_label as-is, others combined to fg_label'''
        assert self.open_world
        out = copy.copy(self)
        traces = {}
        traces[bg_label] = self.get_traces()[bg_label]
        traces[fg_label] = []
        for (domain, its_traces) in self.get_traces().iteritems():
            if domain != bg_label:
                traces[fg_label].extend(its_traces)
        setattr(out, 'traces', traces)
        return out

    @property
    def background(self):
        '''@return if this scenario is a background scenario'''
        return self.path.endswith("@1")

    @property
    def color(self):
        '''@return pre-defined or set color for graphs etc'''
        try:
            return self._color
        except AttributeError:
            return COLORS[self.describe()]

    def set_color(self, color):
        '''sets colors for graphs etc'''
        self._color = color

    @property
    def config(self):
        '''@return configuration of addon if used'''
        if self.name != NEW_DEFENSE:
            return ""
        try:
            fromsettings = INT_REGEXP.search(self.setting).group(1)
        except AttributeError:
            return ""
        try:
            factor = self.status['addon']['factor']
            factor = 50 if factor is None else factor
            assert int(factor) == self.factor2
        except TypeError:
            pass
        return fromsettings

    @property
    def uses_bridge(self):
        '''crude approximation if this scenario uses a bridge'''
        return (hasattr(self, "setting")
                and "bridge" in self.setting
                and "nobridge" not in self.setting)

    def date_from_trace(self):
        '''retrieve date from traces'''
        self.trace_args = {'remove_small': False, 'or_level': 0}
        all_traces = itertools.chain.from_iterable(self.get_traces().values())
        first = min((x.starttime for x in all_traces))
        return first.date()

    def describe(self):
        '''@return name if not new defense, else add config'''
        if not self.config:
            return self.name
        if self.factor2 == 20:
            add = "light"
        elif self.factor2 == 50:
            add = "medium"
        elif self.factor2 == 100:
            add = "heavy"
        else:
            logging.debug("%s from %s", self.factor2, self.setting)
            add = self.config
        return "{} {}".format(self.name, add)

    @property
    def factor2(self):
        '''@return the factor from the settings (if new defense), else ""'''
        try:
            return int(filter(str.isdigit, self.setting))
        except TypeError:
            return int(filter(unicode.isdigit, self.setting))

    # # todo: codup counter.py?
    def get_features_cumul(self, current_sites=True):
        '''@return traces converted to CUMUL's X, y, y_domains'''
        X = []
        out_y = []
        class_number = 0
        domain_names = []
        for domain, dom_counters in self.get_traces(current_sites).iteritems():
            if domain == "background":
                _trace_list_append(X, out_y, domain_names,
                                   dom_counters, "cumul", -1, "background")
            else:
                _trace_list_append(X, out_y, domain_names,
                                   dom_counters, "cumul", class_number, domain)
                class_number += 1
        if len(set(out_y)) == 2:
            out_y = list(mymetrics.binarized(out_y, transform_to=1))
        return (np.array(X), np.array(out_y), domain_names)

    def get_open_world(self, num="auto", same=False, current_sites=True):
        '''
        @return scenario with traces and (num) added background traces
        @param num: size of background set, if 'auto', use as many as fg set
        @param same: only use scenarios of same defense (name, config, site)
        '''
        if self.traces and 'background' in self.get_traces(current_sites):
            logging.warn("scenario's traces already contain background set")
            return self
        filt = None
        if same:
            def filt(x):
                return self._compareattr(x, "name", "config", "site")
        background = self._closest("@1", include_bg=True, filter_scenario=filt)
        logging.info("background is %r", background)
        out = copy.copy(self)
        out.traces = copy.copy(self.get_traces(current_sites))
        if num:
            if num == 'auto':
                num = sum([len(x) for x in self.traces.values()])
            out.traces['background'] = background.get_sample(
                num, current_sites=current_sites)['background']
        else:
            out.traces['background'] = background.get_traces(
                current_sites)['background']
        return out

    def get_sample(self, num, random_seed=None, current_sites=True):
        '''@return sample of traces: each domain has num traces'''
        random.seed(random_seed)
        out = {}
        for (domain, trace_list) in self.get_traces(current_sites).iteritems():
            try:
                out[domain] = random.sample(trace_list, num)
            except ValueError:
                logging.warn("sample size for %s (%s) larger than total (%s)",
                             domain, num, len(trace_list))
                out[domain] = trace_list
        return out

    def get_traces(self, current_sites=True):
        '''@return dict {domain1: [trace1, .., traceN], ..., domainM: [...]}'''
        if not self.traces:
            self.traces = counter.all_from_dir(os.path.join(DIR, self.path),
                                               **self.trace_args)
            if (self._open_world_config
                    and 'current_sites' in self._open_world_config
                    and not self._open_world_config['current_sites']):
                pass
            elif (current_sites
                    or self._open_world_config
                    and 'current_sites' in self._open_world_config
                    and self._open_world_config['current_sites']):
                self.traces = sites.clean(self.traces)
            if self._exclude and isinstance(self._exclude, int):
                num_sites = len(self.traces)
                sorted_sites = self.traces.keys()
                sorted_sites.sort(key=lambda x: sites.cache.index(x))
                self._exclude = sorted_sites[num_sites - self._exclude:]
            for site in self._exclude:
                    try:
                        del self.traces[site]
                    except KeyError:
                        logging.info("failed to exclude %s", site)
            if self._open_world_config:
                self.traces = self.get_open_world(
                    self._open_world_config['background_size'],
                    same=True, current_sites=current_sites).traces
        return self.traces

    @property
    def duration(self):
        '''duration of the capture (time of last - time of first trace)'''
        flat = list(itertools.chain.from_iterable(self.get_traces().values()))
        cmax = max(flat)
        cmin = min(flat)
        return (cmax.starttime
                + datetime.timedelta(0, cmax.duration)
                - cmin.starttime)

    @property
    def num_sites(self):
        '''number of sites in this capture'''
        if hasattr(self, "_num_sites"):
            return self._num_sites
        return len(glob.glob(os.path.join(DIR, self.path, '*json')))

    @property
    def open_world(self):
        '''@return if this scenario has open_world traces'''
        return self.traces and 'background' in self.traces

    @property
    def site(self):
        '''@return the site where this was captured'''
        return utils.site(self.status)

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
        # also without private members (_) etc, used for pd.DataFrame
        return {a: getattr(self, a) for a in dir(self) if (
            not a.startswith('_')
            and not a == 'trace_args'
            and not a == 'traces'
            and not callable(getattr(self, a)))}

    def capture_success(self):
        '''print how successful the capture was'''
        s = Scenario(self.path)
        lsproc = subprocess.Popen(
            "cat *json | wc -l",
            cwd=os.path.join(DIR, s.path),
            stdout=subprocess.PIPE,
            shell=True)
        total_num = int(lsproc.stdout.next())
        out = {"total": total_num}
        print("total: {}".format(total_num))
        s = Scenario(self.path)
        s.trace_args = {
            'or_level': 0, 'remove_small': False, 'remove_timeout': False}
        s._stats_helper("text loaded", total_num, out)
        s = Scenario(self.path)
        s.trace_args = {
            'or_level': 0, 'remove_small': False, 'remove_timeout': True}
        s._stats_helper("fully loaded", total_num, out)
        s = Scenario(self.path)
        s.trace_args = {
            'or_level': 1, 'remove_small': True, 'remove_timeout': False}
        s._stats_helper("or-level 1", total_num, out)
        s = Scenario(self.path)
        s._stats_helper("default", total_num, out)
        print("duration: {}".format(s.duration))
        return out

    def _stats_helper(self, name, total_num, out=None):
        '''print single scenario's count'''
        t = self.get_traces()
        count = sum([len(t[x]) for x in t.keys()])
        if out:
            out[name] = {x: len(t[x]) for x in t.keys()}
        print("{}: {:d} ({:.1f}%)".format(
            name, count, 100. * count / total_num))

def _prepend_if_ends(whole, *parts):
    '''if whole ends with part, prepend it (modulo "-")

    >>> _prepend_if_ends(\
    '0.22/nobridge--2017-01-19-aI-factor=10-with-errors', 'with-errors')
    '0.22/with-errors-nobridge--2017-01-19-aI-factor=10'
    '''
    for part in parts:
        if whole and whole.endswith('-' + part):
            splits = whole.rsplit('/', 1)
            last = part + '-' + splits[1].replace('-' + part, '')
            whole = splits[0] + '/' + last
    return whole


def list_all(in_name=None, include_bg=False, filter_scenario=None, path=DIR):
    '''@return list of all scenarios in =path=.
    @param filter_scenario: function that returns True to include scenarios
    '''
    out = []
    for (dirname, _, _) in os.walk(path):
        if dirname == path:
            continue
        if in_name and in_name not in dirname:
            continue
        out.append(dirname)
    out[:] = [x.replace(path+'/', './') for x in out]
    out = [Scenario(x) for x in _filter_all(
        out, include_bg=include_bg)]
    if filter_scenario:
        out = filter(filter_scenario, out)
    return out


def _filter_all(all_, include_bg):
    '''Filter out specific cases for scenario names,
    @param include_bg if True include background scenarios, else omit'''
    out = [x for x in all_ if (
        '/batch' not in x
        and '/broken' not in x
        and '/features' not in x
        and '/foreground-data' not in x
        and '/or' not in x
        and '/output' not in x
        and '/ow' not in x
        and '/p_batch' not in x
        and '/skip' not in x
        and (include_bg
             or ('/background' not in x
                 and '/bg' not in x
                 and not x.endswith("@1"))))]
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
        logging.warn("keys are different, just used %d common keys", len(keys))
        logging.info("keys: %s", keys)
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

# code duplication
def tpo(trace_list):
    '''returns total outgoing packets for each trace in list'''
    return [x.get_tpo() for x in trace_list]


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
# ## parse older "json" status
# json.loads(b.status.replace("'", '"').replace('False', 'false').replace('u"', '"'))

# ## scenarios without result
# a = {x: len(results.for_scenario(x)) for x in scenario.list_all()}
#  filter(lambda x: x not in ',[]', str([x[0].path for x in filter(lambda x: x[1] == 0, a.iteritems())])) # unfiltered starts with ...[x[0

# ## weird scenario
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

# ### number of LOAD FAILURES PER SITE
# a = scenario.Scenario("defense-client/bridge--2018-01-07--30@50")
# a.trace_args['remove_small'] = False # optional, keeps too-small sites
# ([(name, len(traces)) for (name, traces) in a.get_traces().iteritems()]
# ## proportion of errors
# 1 - np.mean([len(traces) for traces in a.get_traces().values()]) / 50

# ### create TABLE IN EVALUATION, compare results
# ## size overheads
# sorted([x for x in results.list_all() if 'new defense' in x.scenario.name], key=lambda x: abs(x.size_overhead - 163.08) if x.size_overhead else 100000)
# ## score
# sorted([x for x in results.list_all() if 'new defense' in x.scenario.name], key=lambda x: abs(x.score - 0.6822))

# ### all scenarios with possible ow
# import scenario, results
# ow_possible = []
# for scenario_obj in scenario.list_all():
#     if not [r for r in results.for_scenario_open(scenario_obj)]:
#     #        if not r.open_world['binary'] and not r.open_world['auc_bound']]: # this line is optional
#         try:
#             _ = scenario_obj._closest("@1", True, lambda x: scenario_obj._compareattr(x, "name", "config", "site"))
#             ow_possible.append(scenario_obj)
#         except ValueError:
#             pass
