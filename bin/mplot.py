import itertools

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics, model_selection
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
              rotation=45):
    '''plots confusion matrix'''
    confmat = metrics.confusion_matrix(y_true, y_pred)
    domainnames = [x[1] for x in sorted(set(zip(y_true, domains)))]
    df = pd.DataFrame(confmat, index=domainnames, columns=domainnames)
    heatmap = sns.heatmap(df)
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


def _init_roc():
    '''initializes ROC plot'''
    out = plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    return out

def roc(fpr, tpr, title="ROC curve", plot=None):
    '''@return plot object with roc curve, use =.savefig(filename)=
to save, and =.show()= to display.
    @params If =plot=, draw another curve into existing plot'''
    if not plot:
        plot = _init_roc()
    curve = plt.plot(
        fpr, tpr,
        label='{} (AUC = {:0.2f})'.format(title, metrics.auc(fpr, tpr)))
    one_percent = [y for (x,y) in zip(fpr, tpr) if x >= 0.01][0]
    line = plt.plot([0, 1], [one_percent] *2, "red", label='1% fpr')
    #plt.legend([curve, line], loc="lower right") # need to call at the very end
    return plot


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
def traces_cumul(scenario_obj, domain, legend=None, color="red", axes=None):
    X, y, yd = scenario_obj.get_features_cumul()
    data = [x[0] for x in zip(X[:,4:], yd) if x[1] == domain]
    for datum in data:
        line = plt.plot(datum, c=color)
    return line



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
# m = results.for_scenario(scenario.Scenario("disabled/bridge--2016-08-15"))[1]
# X, y, yd = scenario.Scenario("disabled/bridge--2016-08-15").get_features_cumul()
# X = preprocessing.MinMaxScaler().fit_transform(X)
# y_pred = model_selection.cross_val_predict(m.get_classifier(), X, y, cv=config.FOLDS, n_jobs=config.JOBS_NUM)
# confmat, heatmap = mplot.confusion_matrix(y, y_pred, yd, 'Confusion Matrix for disabled/bridge--2016-08-15', rotation=90)
# plt.savefig('/tmp/confmat-2016-08-15.eps')



#(s1, s2, s3) = scenarios.keys()
