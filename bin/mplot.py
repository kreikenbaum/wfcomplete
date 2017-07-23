import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("colorblind") # optional, uglier, but helpful

import scenario


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

''' two plots in one figure with one title etc
reload(mplot)
fig, axes = plt.subplots(2, 1, sharex=True)
mplot.total_packets_in(s, s.keys()[:4], axes[0])
mplot.total_packets_in(s3, s3.keys()[:4], axes[1])
plt.suptitle("Total number of incoming packets")
axes[0].set_xlim(min(min([scenario.tpi(v) for v in s.values()])), max(max([scenario.tpi(v) for v in s3.values()])))
axes[1].set_xlim(min(min([scenario.tpi(v) for v in s.values()])), max(max([scenario.tpi(v) for v in s3.values()])))
axes[0].set_title("no defense")
axes[1].set_title("early defense")
fig.text(0.04, 0.5, "relative histograms with kernel-density-estimation", va="center", rotation="vertical")
plt.savefig("/tmp/total_packets_in_"+'_'.join(s3.keys()[:4])+".pdf")
'''

''' three plots ...
argv = ['', 'disabled/bridge--2016-07-06', 'wtf-pad/bridge--2016-07-05', 'tamaraw']
scenarios = counter.for_scenarios(argv[1:], or_level=2)
fig, axes = plt.subplots(len(scenarios), 1, sharex=True)
plt.suptitle("Number of incoming packets per trace")
mm = counter.MinMaxer()
keys = scenarios.values()[0].keys()
# some chinese sites were problematic via tor
if 'sina.com.cn' in keys: keys.remove('sina.com.cn') 
sitenum = 4
for (i, (name, counter_dict)) in enumerate(scenarios.items()):
    mplot.total_packets_in(counter_dict, keys[:sitenum], axes[i])
    subset = [counter_dict[x] for x in keys[:sitenum]]
    mm.set_if(min(min([scenario.tpi(v) for v in subset])),
              max(max([scenario.tpi(v) for v in subset])))
    axes[i].set_title('scenario: {}'.format(scenario.Scenario(name)))
for (i, _) in enumerate(scenarios):
    axes[i].set_xlim(mm.min, mm.max)
fig.text(0, 0.5, "relative histograms with kernel-density-estimation",
         va="center", rotation="vertical")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("/tmp/total_packets_in_"
            + '_'.join(scenarios).replace('/', '___')+'__'
            +'_'.join(keys[:sitenum])+"__palette_colorblind.pdf")
'''
# todo: outlier removal for tamaraw (see youtube), (?avoid sina.com?)

#(s1, s2, s3) = scenarios.keys()
