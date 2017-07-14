import matplotlib.pyplot as plt
import seaborn as sns

import stats


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
        sns.distplot(stats.tpi(v), label=k, ax=ax)

    if not ax:
        plt.title("Total number of incoming packets")
        plt.xlabel("number of incoming packets")
        plt.ylabel("relative histogram with kernel-density-estimation")
        plt.legend()
    else:
        ax.legend()
    if save:
        plt.savefig("/tmp/total_packets_in_"+'_'.join(subkeys)+".pdf")

''' usage
reload(mplot)
fig, axes = plt.subplots(2, 1, sharex=True)
mplot.total_packets_in(s, s.keys()[:4], axes[0])
mplot.total_packets_in(s3, s3.keys()[:4], axes[1])
plt.suptitle("Total number of incoming packets")
axes[0].set_xlim(min(min([stats.tpi(v) for v in s.values()])), max(max([stats.tpi(v) for v in s3.values()])))
axes[1].set_xlim(min(min([stats.tpi(v) for v in s.values()])), max(max([stats.tpi(v) for v in s3.values()])))
axes[0].set_title("no defense")
axes[1].set_title("early defense")
fig.text(0.04, 0.5, "relative histograms with kernel-density-estimation", va="center", rotation="vertical")
plt.savefig("/tmp/total_packets_in_"+'_'.join(s3.keys()[:4])+".pdf")
'''