#! /usr/bin/env python
'''test lognormal_trunc (extension to scipy if there is time)'''
import unittest

import lognormal_truncated

class TestTruncLognorm(unittest.TestCase):
    # cdf(bigger_top) == 1
    def test_cdf_bigger_top(self):
        t = lognormal_truncated.trunclognorm(1, float('-inf'), 15)
        self.assertAlmostEqual(t.cdf(15.0001), 1, places=4)
        q = lognormal_truncated.trunclognorm(2, float('-inf'), 15)
        self.assertAlmostEqual(q.cdf(15.0001), 1, places=4)
        r = lognormal_truncated.trunclognorm(0.5, float('-inf'), 15)
        self.assertAlmostEqual(r.cdf(15.0001), 1, places=4)

    # cdf(smaller_bottom) == 0
    def test_cdf_smaller_bottom(self):
        t = lognormal_truncated.trunclognorm(1, 1, float('inf'))
        self.assertAlmostEqual(t.cdf(1-0.001), 0)
    def test_cdf_smaller_bottom2(self):
        q = lognormal_truncated.trunclognorm(2, 1, float('inf'))
        self.assertAlmostEqual(q.cdf(1-0.001), 0)
    def test_cdf_smaller_bottom3(self):
        r = lognormal_truncated.trunclognorm(0.5, 1, float('inf'))
        self.assertAlmostEqual(r.cdf(1-0.001), 0)
    def test_cdf_smaller_bottom4(self):
        t = lognormal_truncated.trunclognorm(1, 1, float('inf'), scale=2)
        self.assertAlmostEqual(t.cdf(1-0.001), 0)
    def test_cdf_smaller_bottom5(self):
        t = lognormal_truncated.trunclognorm(1, 1, float('inf'), scale=0.5)
        self.assertAlmostEqual(t.cdf(1-0.001), 0)

# only_top: mean(html_params) = html_mean
# dynamic mean = \int_a^b x g(x) dx / (F(b) - F(a))
# test that default (top=inf, bottom=-inf) falls back with random values for s 
# and add scale

if __name__ == '__main__':
    unittest.main()
