'''plotting methods using matplotlib'''
import datetime
import logging
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import re
import seaborn as sns
# should be external, maybe in analyse?
from sklearn import metrics, model_selection, preprocessing

import config
import counter
import mymetrics
import scenario
import sites
import results

sns.set()  # sns.set_style("darkgrid")
sns.set_palette("colorblind")  # optional, uglier, but helpful
NAMEDATEERR_SPLIT = re.compile('([^@]*)@[0-9]*(.*)')


def _color(name, all_names, palette="colorblind"):
    '''@return color for scatter plot: colors from colorblind palette

    >>> color = lambda x: _color(x, ['a', 'b']); color('a')
    (0.0, 0.4470588235294118, 0.6980392156862745)
    '''
    palette = sns.color_palette(palette, len(all_names))
    for (i, check_name) in enumerate(all_names):
        if name == check_name:
            return palette[i]
    assert False, 'unknown name "{}"'.format(name)


def accuracy_vs_overhead(result_list, title="Size Overhead to Accuracy"):
    '''plots scatter plot of results's accuracy vs overhead'''
    df = pd.DataFrame([r.to_dict() for r in result_list])
    names = sorted(set(df['scenario_name']), reverse=True)  # hack, but enough

    def color(x):
        return scenario.COLORS[x]
    df['color'] = df['scenario_name'].map(color)
    df = df.rename(columns={'size_overhead': 'Size Overhead [%]',
                            'score': 'Accuracy'})
    ax = df.plot.scatter('Size Overhead [%]', 'Accuracy', c=df.color)
    ax.legend(handles=[mpatches.Patch(
        color=color(x), label=x) for x in names])
    ax.set_ybound(0, 1)
    ax.set_title(title)
    plt.tight_layout()
    return ax


# import mplot, results, config, pickle
# r = [r for r in results.list_all() if "2018" in r.scenario and "disabled" in r.scenario and r.open_world][-1]
# config.JOBS_NUM = 3; config.FOLDS = 3; config.VERBOSE = 3
# out = mplot.confusion_matrix_for_result(r)
# pickle.dump(out, file("out.pickle", "w"))
def confusion_matrix_for_result(result, current_sites=True, **kwargs):
    '''creates a confusion matrix plot for result

    @return  confusion_matrix_helper output'''
    try:
        yt, yp, yd = result.y_true, result.y_prediction, result.y_domains
        return (confusion_matrix(
            yt, yp, yd, 'Confusion matrix for {}'.format(result), **kwargs),
                yt, yp, yd)
    except ValueError:
        logging.info("discarded existing prediction: did not match domains")
        return confusion_matrix_helper(
            result.get_classifier(), result.scenario,
            current_sites=current_sites, **kwargs)


def confusion_matrix_from_scenario(scenario_obj, **kwargs):
    '''creates a confusion matrix plot for scenario_obj

    @return confusion_matrix_helper output'''
    r = max(results.for_scenario_smartly(scenario_obj), key=lambda x: x.score)
    return confusion_matrix_for_result(r, **kwargs)


def confusion_matrix_helper(clf, scenario_obj, current_sites=True, **kwargs):
    '''@return (confusion_matrix output, y_true, y_pred, y_domains)'''
    X, y, yd = scenario_obj.get_features_cumul(current_sites)
    X = preprocessing.MinMaxScaler().fit_transform(X)
    y_pred = model_selection.cross_val_predict(
        clf, X, y,
        cv=config.FOLDS, n_jobs=config.JOBS_NUM, verbose=config.VERBOSE)
    return (confusion_matrix(
        y, y_pred, yd,
        'Confusion matrix for {}'.format(scenario_obj), **kwargs),
            y, y_pred, yd)


def confusion_matrix(y_true, y_pred, domains, title='Confusion matrix',
                     rotation=90, normalize=False, number_plot=False,
                     zero0=False):
    '''plots confusion matrix

    @return (confusion matrix dataframe, heatmap plot)'''
    confmat = metrics.confusion_matrix(y_true, y_pred)
    domainnames = [x[1] for x in sorted(set(zip(y_true, domains)))]
    df = pd.DataFrame(confmat, index=domainnames, columns=domainnames)
    df.columns.name = "Prediction"
    df.index.name = "Treatment"
    if zero0:
        df.values[0][0] = 0
    if normalize:
        df = df / df.sum(axis=1)
    heatmap = sns.heatmap(df, annot=number_plot)
    loc, labels = plt.xticks()
    heatmap.set_xticklabels(labels, rotation=rotation)
    heatmap.set_yticklabels(labels, rotation=90-rotation)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return (df, heatmap)



def date_accuracy(size=30):
    '''@return accuracy over time for disabled data of size =size='''
    result_list = [r for r in results.list_all()
                   if r.scenario.num_sites == size
                   and scenario.NODEF in r.scenario.name
                   and "nobridge" not in r.scenario.name
                   and not r.open_world
                   and r.scenario.date < datetime.date(2017, 12, 24)]
    df = pd.DataFrame([x.__dict__ for x in result_list])
    df = df.rename(columns={'score': 'Accuracy'})  # todo: * 100 and ..cy [%]
    df['Date of Capture'] = df['scenario'].map(lambda x: x.date.toordinal())
    plot = df.plot.scatter('Date of Capture', 'Accuracy',
                           color=scenario.COLORS[scenario.NODEF])  # here, too?
    plot.legend(handles=[mpatches.Patch(
        color=scenario.COLORS[scenario.NODEF],
        label=result_list[0].scenario.name)])
    plot.set_title("Accuracy Ratio by Date (on {} sites)".format(size))
    plot.set_ybound(0, 1)
    plot.xaxis.set_major_formatter(mdates.DateFormatter('%m.%Y'))
    plt.tight_layout()
    return plot


def _init_roc(title="ROC curve", titleadd=None):
    '''initializes ROC plot'''
    if titleadd:
        title += " " + titleadd
    out = plt.figure()
    plt.plot([0, 1], [0, 1], 'k--', label="random guessing")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    return out


def roc_helper_open_world_binary(result, **kwargs):
    '''@return fpr-array, tpr-array, figure'''
    assert result.open_world and result.open_world['binary'], "no-owbin result"
    yt = mymetrics.binarize(result.y_true, transform_to=1)
    fpr_array, tpr_array, _ = metrics.roc_curve(
        yt, mymetrics.binarize_probability(result.y_prediction)[:, 1],
        mymetrics.pos_label(yt))
    return fpr_array, tpr_array, roc(
        fpr_array, tpr_array, '{}'.format(result.scenario), **kwargs)


def roc(fpr, tpr, curvelabel="ROC-curve", titleadd=None, fig=None,
        dot_position=0.01, color=None):
    '''@return fig object with roc curve, use =fig.savefig(filename)=
to save, and =fig.show()= to display.
    @param fig: draw another curve into existing figure
    @param dot_position: (0<dot_position<1), draw dot at curve at this fpr'''
    if not fig:
        fig = _init_roc(titleadd=titleadd)
    plt.plot(fpr, tpr, label='{} (AUC = {:0.2f})'.format(
        curvelabel, metrics.auc(fpr, tpr)), color=color)
    if dot_position:
        x1, y1 = [(x, y) for (x, y) in zip(fpr, tpr) if x < dot_position][-1]
        plt.plot(x1, y1, "ro",
                 label='{:2.2f}% false-, {:2.2f}% true positives'.format(
                     x1*100, y1*100))
    plt.legend()
    fig.get_axes()[0].set_ybound(-0.01, 1.01)
    fig.get_axes()[0].set_xbound(-0.01, 1.01)
    # plt.tight_layout()
    return fig


# should be rename to sth general
def total_packets_in(counter_dict, subkeys=None, ax=None, save=False,
                     color=None, method=scenario.tpi):
    '''plots total incoming packets (see method) stat, rugplot with kde

    - plot size histogram colored by domain
      - with kde
    Usage:
    total_packets_in(scenarios.values()[0], scenarios.values()[0].keys()[:4])
    '''
    plt.xscale("log")
    if not subkeys:
        subkeys = counter_dict.keys()
    for domain in subkeys:
        #        sns.distplot(stats.tpi(v), hist=False, rug=True, label=k)
        c = color(domain) if color else None
        sns.distplot(method(counter_dict[domain]), label=domain,
                     ax=ax, color=c)

    if not ax:
        plt.title("Total number of incoming packets")
        plt.xlabel("number of incoming packets")
        plt.ylabel("relative count with kernel-density-estimation")
        plt.legend()
    else:
        ax.legend()
    if save:
        plt.savefig("/tmp/total_packets_in_"+'_'.join(subkeys)+".pdf")


def total_packets_in_site(scenarios, site):
    '''plot of site's total number of incoming packets for each scenario'''
    ARGS = {'or_level': 0, 'remove_timeout': True, 'remove_small': False}
    for s in scenarios:
        if s.trace_args != ARGS:
            s.traces = None
            s.trace_args = ARGS
        sns.distplot(scenario.tpi(s.get_traces()[site]),
                     label=s.describe(), color=s.color)
    plt.legend()


def total_packets_in_helper(names, trace_dicts=None, sitenum=4, save=True):
    '''plot tpi plots in subplots

    example input:
    names = ['disabled/bridge--2016-07-06', 'wtf-pad/bridge--2016-07-05']
    names = {x.path: x.get_traces() for x in scenario_list}
    '''
    if not trace_dicts:
        trace_dicts = [scenario.Scenario(name).get_traces() for name in names]
        fig, axes = plt.subplots(len(names), 1, sharex=True, sharey=False)
    plt.suptitle("Number of incoming packets per trace")
    mm = counter.MinMaxer()
    keys = set(trace_dicts[0].keys())
    if 'sina.com.cn' in keys:
        keys.remove('sina.com.cn')
    for other_dict in trace_dicts[1:]:
        keys = keys.intersection(other_dict.keys())
    keys = sorted(keys, key=lambda x: sites.cache.index(x))[:sitenum]

    def color(x):
        return _color(x, keys)
    for (name, counter_dict, ax) in zip(names, trace_dicts, axes):
        total_packets_in(counter_dict, keys, ax, color=color)
        subset = [counter_dict[x] for x in keys]
        mm.set_if(min(min([scenario.tpi(v) for v in subset])),
                  max(max([scenario.tpi(v) for v in subset])))
        ax.set_title('{}'.format(scenario.Scenario(name).describe()))
    for ax in axes:
        ax.set_xlim(mm.min * 0.8, mm.max * 1.2)
    fig.text(0, 0.5, "relative histograms with kernel-density-estimation",
             va="center", rotation="vertical")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save:
        plt.savefig(
            str("/tmp/total_packets_in_" + '_'.join(names).replace('/', '___')
                + '__' + '_'.join(keys) + "__palette_colorblind")[:250]
            + ".pdf")
    return trace_dicts


# # traces_cumul usage (two sites):
# s = scenario.list_all("2017-12-31")[0]
# a = ['wikipedia.org', 'onclickads.net']
# color = lambda x: mplot._color(x, a)
# ax = mplot.plt.axes()
# for domain in a: mplot.traces_cumul(s, domain, color(domain), axes=ax)
# mplot.plt.legend()
# mplot.plt.xlabel("Feature Index")
# mplot.plt.ylabel("Feature Value [Byte]")
# mplot.plt.title("CUMUL example for two sites retrieved on {}".format(s.date))
# mplot.plt.tight_layout()
def traces_cumul(scenario_obj, domain, color=sns.color_palette()[0],
                 save=False, axes=None, current_sites=True):
    '''plots the cumul traces of =domain= in scenario_obj'''
    X, y, yd = scenario_obj.get_features_cumul(current_sites)
    data = [x[0] for x in zip(X, yd) if x[1] == domain]  # zip(X[:, 4:]
    legend = domain
    if not axes:
        _, axes = plt.subplots()
    for datum in data:
        # line = plt.plot(datum, c=color, alpha=0.5, linewidth=1, axes=axes,
        #                 label=legend)
        line = axes.plot(datum, c=color, alpha=0.5, linewidth=1, label=legend)
        legend = None
    axes.set_title("Traces for {} captured with {}".format(domain,
                                                           scenario_obj))
    if save:
        plt.savefig("/tmp/traces_cumul_{}_{}.pdf".format(scenario_obj, domain)
                    .replace(" ", "_"))
    return line


def _splitdate(trace_name):
    '''@return trace's name + possible error cause, splits name'''
    return ''.join(NAMEDATEERR_SPLIT.search(trace_name).groups())



def ccdf_curve_for_scenario(scenario_obj, existing=True, axes=None,
                            filt=None, type_="recall", save=False):
    if not axes:
        _, axes = plt.subplots()
    used = set()
    for result in sorted((r for r in results.for_scenario_open(scenario_obj)
                         if not r.open_world['binary']),
                         key=lambda r: r.background_size):
        logging.debug(result)
        if (result.open_world['tpr'] == 0.0 and result.open_world['fpr'] == 0.0
            or result.open_world['tpr'] <= 0.01
            or scenario_obj.num_sites - result.size > 10):
            logging.debug("skipped %s", result)
            continue
        if len(used) >= 6:
            break
        if (existing and not (result._ypred and result._ytrue) or
                filt and not filt(result)):
            logging.debug("skipped %s", result)
            continue
        # size = len(result.scenario.get_traces()['background'])
        if result.background_size in (r.background_size for r in used):
            continue
        used.add(result)
    for result in sorted(used, key=lambda x: int(x.background_size)):
        ccdf_curve(result.get_confusion_matrix(), result.background_size,
                   # "{} ({})".format(size, result._id),
                   axes=axes, type_=type_)
    plt.legend()
    plt.title("CCDF-{} curve for {}".format(type_, scenario_obj))
    if save:
        plt.savefig("/tmp/ccdf-{}-{}.pdf"
                    .format(type_, scenario_obj)
                    .replace(" ", "-"))
    return used


def ccdf_curve_for_results(results, type_="recall", axes=None, **kwargs):
    for result in results:
        ccdf_curve(result.get_confusion_matrix(),
                   len(result.scenario.traces['background']),
                   type_=type_, axes=axes, **kwargs)


METRIC = {"recall": 0, "false-positive rate": 1, "precision": 2}


def ccdf_curve(confmat, bg_size, step=0.1, type_="recall", axes=None):
    '''plots ccdf curve as in CUMUL paper Fig.8'''
    if not axes:
        _, axes = plt.subplots()
    plotx = np.arange(0, 1.01, step=step)
    ploty = []
    tpr_fpr_tpa = mymetrics.tpr_fpr_tpa(confmat)
    for xpos in plotx:
        ploty.append(len([x for x in tpr_fpr_tpa if x[METRIC[type_]] > xpos])
                     * 100. / len(confmat))
    _curve(type_, plotx, ploty, bg_size, "b = {}".format(bg_size), axes)


def _curve(name, x, y, bg_size, label, axes):
    axes.plot(x, y, marker="o", label="b = {}".format(bg_size))
    axes.set_xlabel(name)
    axes.set_ylabel("Fraction of Foreground Pages [%]")

# # usage:
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
    # X = [x.cumul()[4:] for x in traces]
    X = [x.cumul() for x in traces]
    label = _splitdate(traces[0].name)
    for datum in X:
        plt.plot(datum, c=color, alpha=.5, linewidth=1, axes=axes, label=label)
        label = None

# # PLOT ROWS OF TRACES (mostly related chosen)
# a = scenario.list_all("17-12-26")[0]
# wiki = a.get_traces()["wikipedia.org"]
# _, g = plt.subplots(7, 1)
# for i, el in enumerate([wiki[x-1] for x in [5, 26, 45, 35, 32, 24, 44]]):#, 1]]):
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
        if idx == len(times)-1:
            continue
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

# ## USAGE EXAMPLES
# # Confusion Matrix from Scenario
# r = results.for_scenario(scenario.Scenario("disabled/bridge--2016-08-15"))[1]
# X, y, yd = scenario.Scenario("disabled/bridge--2016-08-15").get_features_cumul()
# X = preprocessing.MinMaxScaler().fit_transform(X)
# y_pred = model_selection.cross_val_predict(r.get_classifier(), X, y, cv=config.FOLDS, n_jobs=config.JOBS_NUM)
# confmat, heatmap = mplot.confusion_matrix(y, y_pred, yd, 'Confusion Matrix for disabled/bridge--2016-08-15', rotation=90)
# plt.savefig('/tmp/confmat-2016-08-15.eps')



#(s1, s2, s3) = scenarios.keys()
