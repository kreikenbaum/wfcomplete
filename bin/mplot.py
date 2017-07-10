import matplotlib.pyplot as plt
import seaborn as sns


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


def total_packets_in(counter_dict):
    '''plots total incoming packets stat, absolute and with kde

    - plot size histogram colored by domain
      - with kde
    '''

'''
0. [@0] take 10 old sites
   - [later: take those combined with 100-defenses for difference]
   - load via
1. compute =totalnumber_in= mean and std for each element
   0. [@0] =total_number_in= as of danezis
   1. use panchenko 1 or 2 (better 1, labeled)'s extraction to
      get per-trace
   2. compute stats? or just use seaborn?
2. plot with different colors, see if 10 ok, or needs less
'''    
