import itertools

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
# should be external, maybe in analyse?
from sklearn import metrics, model_selection, preprocessing
sns.set_palette("colorblind") # optional, uglier, but helpful

import config
import scenario
import results


def _color(name, all_names):
    '''@return color for scatter plot: colors from colorblind palette

    >>> color = lambda x: _color(x, ['a', 'b']); color('a')
    (0.0, 0.4470588235294118, 0.6980392156862745)
    '''
    palette = sns.color_palette("colorblind", len(all_names))
    for (i, check_name) in enumerate(all_names):
        if name == check_name:
            return palette[i]
    else:
        assert 'wtf (what a terrible failure)'


def accuracy_vs_overhead(result_list, title="Size Overhead to Accuracy"):
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


# parts due to sklearn's plot_confusion_matrix.py
def confusion_matrix(y_true, y_pred, domains, title='Confusion matrix',
                     rotation=90, normalize=False, number_plot=False):
    '''plots confusion matrix'''
    confmat = metrics.confusion_matrix(y_true, y_pred)
    domainnames = [x[1] for x in sorted(set(zip(y_true, domains)))]
    df = pd.DataFrame(confmat, index=domainnames, columns=domainnames)
    heatmap = sns.heatmap(df)
    if normalize:
        confmat = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis] * 100
        confmat = confmat.astype('int')
    if number_plot:
        thresh = confmat.max() / 2.
        for i, j in itertools.product(range(confmat.shape[0]),
                                      range(confmat.shape[1])):
            plt.text(confmat.shape[1]-j-1, i+1, confmat[i, j],
                     horizontalalignment="left",
                     verticalalignment="top",
                     color="white" if confmat[i, j] > thresh else "black")
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
    scenarios = [x for x in list_all() if x.scenario.num_sites == size and 'disabled' in x.scenario.name]
    df = pd.DataFrame([x.__dict__ for x in scenarios])
    df = df.rename(columns={'score': 'Accuracy [%]'})
    df['Scenario Date [ordinal]'] = df['scenario'].map(
        lambda x: x.date.toordinal())
    plot = df.plot.scatter('Scenario Date [ordinal]', 'Accuracy [%]')
    plot.legend(handles=[mpatches.Patch(color=sns.color_palette("colorblind", 1)[0], label='defenseless')])
    plot.set_title("Accuracy Ratio by Date (on {} sites)".format(size))
    plot.set_ybound(0, 1)
    plt.tight_layout()
    return plot


def _init_roc(titleadd = None):
    '''initializes ROC plot'''
    title = "ROC curve"
    if titleadd:
        title += " " + titleadd
    out = plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    return out

def roc_helper(result, axes=None):
    scenario_obj = result.scenario.get_open_world(
        result.open_world['background_size']).binarize()
    X, y, d = scenario_obj.get_features_cumul()
    X = preprocessing.MinMaxScaler().fit_transform(X) # scaling is idempotent
    y_pred = model_selection.cross_val_predict(
        result.get_classifier(True), X, y,
        cv=config.FOLDS, n_jobs=config.JOBS_NUM, method="predict_proba")
    fpr_array, tpr_array, _ = metrics.roc_curve(
        y, y_pred[:, 1], y[np.where(y != -1)[0][0]])
    return roc(
        fpr_array, tpr_array,
        ', max_tpr: {}, background_size: {}'.format(
            result.open_world['auc_bound'],
            result.open_world['background_size'], axes))

def roc(fpr, tpr, titleadd=None, fig=None):
    '''@return fig object with roc curve, use =.savefig(filename)=
to save, and =.show()= to display.
    @params If =fig=, draw another curve into existing figure'''
    if not fig:
        fig = _init_roc(titleadd)
    curve = plt.plot(
        fpr, tpr)
        #label='{} (AUC = {:0.2f})'.format(title, metrics.auc(fpr, tpr)))
    one_percent = [y for (x,y) in zip(fpr, tpr) if x >= 0.01][0]
    line = plt.plot([0, 1], [one_percent] *2, "red", label='1% fpr')
    # plt.legend([curve, line], loc="lower right") # todo: show legend
    fig.get_axes()[0].set_ybound(0, 1)
    plt.tight_layout()
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
## usage:
# a = sorted(scenario.list_all(), key=lambda x: x.date if hasattr(x, "date") else datetime.date(1970, 1, 1))[-2]
# ax = plt.axes()
# traces_cumul(a, 'paypal.com', color="red", axes=ax)
# traces_cumul(a, 'msn.com', color="blue", axes=ax)
# plt.legend()
def traces_cumul(scenario_obj, domain, color="red", axes=None):
    X, y, yd = scenario_obj.get_features_cumul()
    data = [x[0] for x in zip(X[:,4:], yd) if x[1] == domain]
    legend = domain
    for datum in data:
        line = plt.plot(datum, c=color, alpha=0.5, linewidth=1, axes=axes, label=legend)
        legend = None
    return line

## plot row of traces (mostly related chosen)
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
