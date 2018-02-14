'''plotting methods using matplotlib'''
import itertools
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
# should be external, maybe in analyse?
from sklearn import metrics, model_selection, preprocessing
sns.set() #sns.set_style("darkgrid")
sns.set_palette("colorblind") # optional, uglier, but helpful

import config
import mymetrics
import scenario
import results

def _color(name, all_names, palette="colorblind"):
    '''@return color for scatter plot: colors from colorblind palette

    >>> color = lambda x: _color(x, ['a', 'b']); color('a')
    (0.0, 0.4470588235294118, 0.6980392156862745)
    '''
    palette = sns.color_palette(palette, len(all_names))
    for (i, check_name) in enumerate(all_names):
        if name == check_name:
            return palette[i]
    assert 'wtf (what a terrible failure)'


def accuracy_vs_overhead(result_list, title="Size Overhead to Accuracy"):
    '''plots scatter plot of results's accuracy vs overhead'''
    df = pd.DataFrame([x.__dict__ for x in result_list])
    names = set([x.name for x in df['scenario']])
    color = lambda x: _color(x.name, names)
    df['color'] = df['scenario'].map(color)
    df = df.rename(columns={'size_overhead': 'Size Overhead [%]',
                            'score': 'Accuracy'})
    plot = df.plot.scatter('Size Overhead [%]', 'Accuracy', c=df.color)
    plot.legend(handles=[mpatches.Patch(
        color=color(scenario.Scenario(x)),
        label=str(scenario.Scenario(x))) for x in names])
    plot.set_ybound(0, 1)
    plot.set_title(title)
    plt.tight_layout()
    return plot


def confusion_matrix_helper(scenario_obj, **kwargs):
    '''creates a confusion matrix plot for scenario_obj'''
    X, y, yd = scenario_obj.get_features_cumul()
    X = preprocessing.MinMaxScaler().fit_transform(X)
    r = max(results.for_scenario_smartly(scenario_obj), key=lambda x: x.score)
    y_pred = model_selection.cross_val_predict(
        r.get_classifier(), X, y, cv=config.FOLDS, n_jobs=config.JOBS_NUM)
    return confusion_matrix(
        y, y_pred, yd, 'Confusion matrix for {}'.format(scenario_obj), **kwargs)

def confusion_matrix(y_true, y_pred, domains, title='Confusion matrix',
                     rotation=90, normalize=False, number_plot=False):
    '''plots confusion matrix'''
    confmat = metrics.confusion_matrix(y_true, y_pred)
    domainnames = [x[1] for x in sorted(set(zip(y_true, domains)))]
    df = pd.DataFrame(confmat, index=domainnames, columns=domainnames)
    if normalize:
        df = df / df.sum(axis=1)
    heatmap = sns.heatmap(df, annot=number_plot)
    loc, labels = plt.xticks()
    heatmap.set_xticklabels(labels, rotation=rotation)
    heatmap.set_yticklabels(labels[::-1], rotation=90-rotation)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return (confmat, heatmap)


def date_accuracy(size=30):
    '''@return accuracy over time for disabled data of size =size='''
    scenarios = [x for x in results.list_all() if x.scenario.num_sites == size and 'no defense' in x.scenario.name]
    df = pd.DataFrame([x.__dict__ for x in scenarios])
    df = df.rename(columns={'score': 'Accuracy'}) # todo: * 100 and ..cy [%]
    df['Scenario Date [ordinal]'] = df['scenario'].map(
        lambda x: x.date.toordinal())
    plot = df.plot.scatter('Scenario Date [ordinal]', 'Accuracy') # todo2:here2
    plot.legend(handles=[mpatches.Patch(color=sns.color_palette("colorblind", 1)[0], label=scenarios[0].name)])
    plot.set_title("Accuracy Ratio by Date (on {} sites)".format(size))
    plot.set_ybound(0, 1)
    plt.tight_layout()
    return plot


def _init_roc(titleadd=None):
    '''initializes ROC plot'''
    title = "ROC curve"
    if titleadd:
        title += " " + titleadd
    out = plt.figure()
    plt.plot([0, 1], [0, 1], 'k--', label="random guessing")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    return out

def roc_helper(result, axes=None):
    assert result.open_world and result.open_world['binary'], "non-owbin result"
    num = result.open_world['background_size']
    auc_bound = result.open_world['auc_bound']
    scenario_obj = result.scenario.get_open_world(num).binarize()
    X, y, _ = scenario_obj.get_features_cumul()
    X = preprocessing.MinMaxScaler().fit_transform(X) # scaling is idempotent
    y_pred = model_selection.cross_val_predict(
        result.get_classifier(True), X, y,
        cv=config.FOLDS, n_jobs=config.JOBS_NUM, method="predict_proba")
    fpr_array, tpr_array, _ = metrics.roc_curve(
        y, y_pred[:, 1], mymetrics.pos_label(y))
    return fpr_array, tpr_array, roc(fpr_array, tpr_array,
               '({}), max_fpr: {}, background_size: {}'.format(
                   scenario_obj, auc_bound, num, axes))

def roc(fpr, tpr, titleadd=None, fig=None, dot=0.01):
    '''@return fig object with roc curve, use =.savefig(filename)=
to save, and =.show()= to display.
    @params If =fig=, draw another curve into existing figure
        if dot > 0, draw a dot at this fpr rate (<1)'''
    if not fig:
        fig = _init_roc(titleadd)
    curve = plt.plot(
        fpr, tpr,
        label='{} (AUC = {:0.2f})'.format("ROC-curve", metrics.auc(fpr, tpr)))
    #one_percent = [y for (x, y) in zip(fpr, tpr) if x >= 0.01][0]
    #line = plt.plot([0, 1], [one_percent] *2, "red", label='1% false positives')
    if dot:
        x1, y1 = [(x,y) for (x, y) in zip(fpr, tpr) if x < dot][-1]
        plt.plot(x1, y1, "ro",
                 label='{:2.2f}% false-, {:2.2f}% true positives'.format(
                     x1*100, y1*100))
    plt.legend()
    fig.get_axes()[0].set_ybound(-0.01, 1.01)
    fig.get_axes()[0].set_xbound(-0.01, 1.01)
    # plt.tight_layout()
    return fig


def total_packets_in(counter_dict, subkeys=None, ax=None, save=False):
    '''plots total incoming packets stat, rugplot with kde

    - plot size histogram colored by domain
      - with kde
    Usage:
       total_packets_in(scenarios.values()[0], scenarios.values()[0].keys()[:4])
    '''
    plt.xscale("log")
    if not subkeys: subkeys = counter_dict.keys()
    for (k, v) in counter_dict.iteritems():
        if k not in subkeys:
            continue
        #        sns.distplot(stats.tpi(v), hist=False, rug=True, label=k)
        sns.distplot(scenario.tpi(v), label=k, ax=ax)

    if not ax:
        plt.title("Total number of incoming packets")
        plt.xlabel("number of incoming packets")
        plt.ylabel("relative histogram with kernel-density-estimation")
        plt.legend()
    else:
        ax.legend()
    if save:
        plt.savefig("/tmp/total_packets_in_"+'_'.join(subkeys)+".pdf")


# td: color
'''best way (off of head)
- load as pandas dataframe data
- data.T.plot(kind='bar')'''

'''other ways
- plt.bar
  - needs to set lots of options to look good, even with seaborn'''
## traces_cumul usage:
# s = scenario.list_all("2017-12-31")[0]
# a = ['wikipedia.org', 'onclickads.net']
# color = lambda x: mplot._color(x, a)
# ax = mplot.plt.axes()
# for domain in a: mplot.traces_cumul(s, domain, color(domain), ax)
# mplot.plt.legend()
# mplot.plt.xlabel("Feature Index")
# mplot.plt.ylabel("Feature Value [Byte]")
# mplot.plt.title("CUMUL example for two sites retrieved on {}".format(s.date))
# mplot.plt.tight_layout()
def traces_cumul(scenario_obj, domain, color="red", axes=None):
    X, y, yd = scenario_obj.get_features_cumul()
    #data = [x[0] for x in zip(X[:, 4:], yd) if x[1] == domain]
    data = [x[0] for x in zip(X, yd) if x[1] == domain]
    legend = domain
    for datum in data:
        line = plt.plot(datum, c=color, alpha=0.5, linewidth=1, axes=axes, label=legend)
        legend = None
    return line

NAMEDATEERR = re.compile('([^@]*)@[0-9]*(.*)')
def _splitdate(trace_name):
    '''returns trace's name + possible error cause, splits name'''
    return ''.join(NAMEDATEERR.search(trace_name).groups())

## usage:
# s = scenario.list_all("17-12-26")[0]
# t = s.get_traces()['twitter.com']
# terr = [x for x in t if 'Error_loading' in x.name]
# tok = [x for x in t if 'Error_loading' not in x.name]
# ya = s.get_traces()['yandex.ru']
# a = [terr, tok, ya]
# color = lambda x: mplot._color(x, a, "husl")
# for el in a: mplot.traces_cumul_group(el, color(el))
# plt.xlabel("Feature Index")
# plt.ylabel("Feature Value [Byte]")
# plt.tight_layout()
# plt.legend()
def traces_cumul_group(traces, color="red", axes=None):
    #X = [x.cumul()[4:] for x in traces]
    X = [x.cumul() for x in traces]
    label = _splitdate(traces[0].name)
    for datum in X:
        plt.plot(datum, c=color, alpha=0.5, linewidth=1, axes=axes, label=label)
        label = None

## PLOT ROWS OF TRACES (mostly related chosen)
#a = scenario.list_all("17-12-26")[0]
#wiki = a.get_traces()["wikipedia.org"]
#_, g = plt.subplots(7, 1)
#for i, el in enumerate([wiki[x-1] for x in [5, 26, 45, 35, 32, 24, 44]]):#, 1]]):
#    mplot.trace(el, g[i])
def trace(trace, axes=None, color=None):
    '''plot single trace (index-st of domain) as kinda-boxplot'''
    if not axes:
        axes = plt.axes()
    # axes.vlines([x[0] for x in trace.timing], 0, [x[1] for x in trace.timing])
    # axes.stem([x[0] for x in trace.timing], [x[1] for x in trace.timing])
    # axes.step([x[0] for x in trace.timing], [x[1] for x in trace.timing],
    #           where='post', color=color)
    times = [x[0] for x in trace.timing]
    widths = []
    heights = [x[1] for x in trace.timing]
    for idx, el in enumerate(times):
        if idx == len(times)-1: continue
        widths.append(times[idx+1] - times[idx])
    widths.append(np.mean(widths))
    axes.bar(times, heights, widths, fill=False, align="edge", edgecolor=color)


def traces(traces_list):
    '''plots traces on subplots'''
    _, g = plt.subplots(len(traces_list), 1, sharex=True)
    for i, el in enumerate(traces_list):
        trace(el, g[i])
    plt.suptitle("WF Example Traces wikipedia.org")
    g[4].set_ylabel("packet size [Byte]")
    g[-1].set_xlabel("seconds")

### usage examples

## total_packets_in: three plots
# argv = ['', 'disabled/bridge--2016-07-06', 'wtf-pad/bridge--2016-07-05', 'tamaraw']
# #scenarios = counter.for_scenarios(argv[1:], or_level=2)
# scenarios = {x: scenario.Scenario(x, trace_args={'or_level':2}).get_traces() for x in argv[1:]}
# fig, axes = plt.subplots(len(scenarios), 1, sharex=True)
# plt.suptitle("Number of incoming packets per trace")
# mm = counter.MinMaxer()
# keys = scenarios.values()[0].keys()
# # some chinese sites were problematic via tor
# if 'sina.com.cn' in keys: keys.remove('sina.com.cn')
# sitenum = 4
# for (i, (name, counter_dict)) in enumerate(scenarios.items()):
#     mplot.total_packets_in(counter_dict, keys[:sitenum], axes[i])
#     subset = [counter_dict[x] for x in keys[:sitenum]]
#     mm.set_if(min(min([scenario.tpi(v) for v in subset])),
#               max(max([scenario.tpi(v) for v in subset])))
#     axes[i].set_title('scenario: {}'.format(scenario.Scenario(name)))
# for (i, _) in enumerate(scenarios):
#     axes[i].set_xlim(mm.min, mm.max)
# fig.text(0, 0.5, "relative histograms with kernel-density-estimation",
#          va="center", rotation="vertical")
# fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig("/tmp/total_packets_in_"
#             + '_'.join(scenarios).replace('/', '___')+'__'
#             +'_'.join(keys[:sitenum])+"__palette_colorblind.pdf")

## Confusion Matrix from Scenario
# r = results.for_scenario(scenario.Scenario("disabled/bridge--2016-08-15"))[1]
# X, y, yd = scenario.Scenario("disabled/bridge--2016-08-15").get_features_cumul()
# X = preprocessing.MinMaxScaler().fit_transform(X)
# y_pred = model_selection.cross_val_predict(r.get_classifier(), X, y, cv=config.FOLDS, n_jobs=config.JOBS_NUM)
# confmat, heatmap = mplot.confusion_matrix(y, y_pred, yd, 'Confusion Matrix for disabled/bridge--2016-08-15', rotation=90)
# plt.savefig('/tmp/confmat-2016-08-15.eps')



#(s1, s2, s3) = scenarios.keys()
