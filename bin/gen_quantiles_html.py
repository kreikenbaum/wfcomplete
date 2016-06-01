#! /usr/bin/env python
'''gen quantiles of html-lognormal distribution (params given by traffic model'''
import math
from scipy import stats

sigma = 1.7643
mu = 7.90272

parts = 3

html = stats.lognorm(s=sigma, scale=math.exp(mu))
for i in range(2*parts):
    q = float(i) / (2*parts)
    print 'at {}: value {}'.format(q, html.ppf(q))
