#! /usr/bin/env python
'''tests all of scipy's distributions against data set, sees which fits best'''

import doctest
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import _continuous_distns as cd

# if you import by hand, include the path for the counter-module via
# import sys; sys.path.append('/home/w00k/da/git/bin')
# import sys; sys.path.append('/home/mkreik/bin')
import counter

#LOGLEVEL = logging.DEBUG
LOGLEVEL = logging.INFO
#LOGLEVEL = logging.WARN
TIME_SEPARATOR = '@'

def show_hist(counters, feature_name='html_marker'):
    '''show histogram of feature_name'''
    all_dom = []
    labels = []
    for domain in counters.keys():
        all_dom.append([x.get() for x in counters[domain]])
        labels.append(domain)
    plt.hist(all_dom, label=labels, histtype='barstacked', bins=100)
    plt.show()
#    plt.hist(all_dom, label=labels, histtype='barstacked', bins=np.logspace(0, 6, base=10, num=15))

def unused():
    '''unused code'''
    # show hist
    for dom in all_dom:
        plt.hist(dom, bins=100)
        plt.show()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for i in counters['msn.com'][0].fixed.keys():
        print i
        all_dom = []
        labels = []
        for domain in counters.keys():
            all_dom.append([x.get(i) for x in counters[domain]])
            labels.append(domain)
        plt.hist(all_dom, label=labels, histtype='barstacked', bins=100)
        fname = '/mnt/data/' + i + '.png'
        print fname
        plt.savefig(fname)

if __name__ == "__main__":
    doctest.testmod()
    logging.basicConfig(format='%(levelname)s:%(message)s', level=LOGLEVEL)

    # if by hand: change to the right directory before importing
    # os.chdir('/home/w00k/da/sw/data/json/part')
    # os.chdir('/mnt/data/2-top100dupremoved_cleaned/')
    with counter.Counter.from_('/home/w00k/da/sw/data/json/part') as c:
        all_dom = []
        for domain in c.keys():
            all_dom.extend([x.get('html_marker') for x in c[domain]])

    # idee: tests mean, var ==
stats.alpha, stats.anglit, stats.arcsine, stats.beta, stats.betaprime, stats.bradford, stats.burr, stats.cauchy, stats.chi, stats.chi2, stats.cosine, stats.dgamma, stats.dweibull, stats.erlang, stats.expon, stats.exponweib, stats.exponpow, stats.f, stats.fatiguelife, stats.fisk, stats.foldcauchy, stats.foldnorm, stats.frechet_r, stats.frechet_l, stats.genlogistic, stats.genpareto, stats.genexpon, stats.genextreme, stats.gausshyper, stats.gamma, stats.gengamma, stats.genhalflogistic, stats.gilbrat, stats.gompertz, stats.gumbel_r, stats.gumbel_l, stats.halfcauchy, stats.halflogistic, stats.halfnorm, stats.hypsecant, stats.invgamma, stats.invgauss, stats.invweibull, stats.johnsonsb, stats.johnsonsu, stats.ksone, stats.kstwobign, stats.laplace , stats.logistic, stats.loggamma, stats.loglaplace, stats.lognorm , stats.lomax, stats.maxwell , stats.mielke, stats.nakagami, stats.ncx2, stats.ncf, stats.nct, stats.norm, stats.pareto, stats.pearson3, stats.powerlaw, stats.powerlognorm, stats.powernorm, stats.rdist, stats.reciprocal, stats.rayleigh, stats.rice, stats.recipinvgauss, stats.semicircular, stats.t, stats.triang, stats.truncexpon, stats.truncnorm, stats.tukeylambda, stats.uniform , stats.vonmises, stats.wald, stats.weibull_min, stats.weibull_max, stats.wrapcauchy


    



    
([x for dom in all_dom  for x in dom])
scipy.stats.cauchy.fit([x for dom in all_dom  for x in dom])
