#! /usr/bin/env python
'''gen quantiles of number-of-embedded-gamma distribution (params
given by traffic model'''
import math
from scipy import stats

PARTS = 5

kappa=0.141385
theta=40.3257
# must have mean of kappa*theta, which works, which is 5.7

num_embedded = stats.gamma(kappa, scale=theta)
for i in range(2*PARTS):
    q = float(i) / (2*PARTS)
    print 'at {}: value {}'.format(q, num_embedded.ppf(q))


