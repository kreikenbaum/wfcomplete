from scipy import stats
import logging
import math

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

class trunclognorm_gen(stats.rv_continuous):
     """A truncated lognormal continuous random variable.

     The probability density function for `trunclognorm` is::

         l = lognorm(s=sigma, scale=math.exp(mu))
         trunclognorm(x, mu, sigma, bottom, top) 
             = l.pdf(x) / (l.cdf(top) -l.cdf(bottom))

     for ''bottom < x < top''.

     `trunclognorm` takes ``bottom`` and ``top`` as additional
     parameters.  bottom=float('-inf') and top=float('inf') falls back
     to the lognormal distribution.

     """
     def _argcheck(self, s, bottom, top):
          self.bottom = bottom or float('-inf')
          self.top = top or float('inf')
          return (bottom < top) and s > 0

     def _pdf(self, x, s, bottom, top):
         l = stats.lognorm(s)
         if x <= bottom or x >= top:
              logging.debug('outside bounds: %f', x)
              return 0
         else:
              return l.pdf(x) / (l.cdf(top) -l.cdf(bottom))

     def _cdf(self, x, s, bottom, top):
          '''\int_x^a g(t) dt / (F(b) - F(a))
          = G(x) - G(a) / (F(b) - F(a))
          = G(x) -F(bottom) - G(a) +F(bottom) / (F(b) - F(a))
          = (G(x) -F(bottom)) - (G(a) -F(bottom)) / (F(b) - F(a))
          =? F(x) - F(a) / (F(b) - F(a))'''
         l = stats.lognorm(s)

trunclognorm = trunclognorm_gen(name='trunclognorm')

class html_gen(stats.rv_continuous):
     """A trunlognorm variable which samples html sizes."""
     SIGMA = 1.7643
     MU = 7.90272
     TOP = 2 * 1024 * 1024

     def _pdf(self, x):
         l = stats.lognorm(s=self.SIGMA, scale=math.exp(self.MU))
         if x >= self.TOP:
              return 0
         else:
              return l.pdf(x) / l.cdf(self.TOP)

html = html_gen(name='html')
