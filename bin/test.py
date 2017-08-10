#! /usr/bin/env python
'''unit tests counter, analyse and fix modules'''
import datetime
import doctest
import logging
import os
import tempfile
import unittest
import numpy as np

import analyse
import counter
import fit
import scenario

fit.FOLDS = 2


class TestCounter(unittest.TestCase):
    '''tests the counter module'''

    def setUp(self):
        self.c_list = [counter._test(x) for x in [1, 2, 2, 2, 2, 3, 4]] # len 7
        self.big_val = [counter._test(3, val=10*60*1000)] # very big, but ok


    def test_doc(self):
        (fail_num, _) = doctest.testmod(counter, optionflags=doctest.ELLIPSIS)
        self.assertEqual(0, fail_num)


    def test__test(self):
        c = counter._test(35)
        self.assertTrue(c.timing)
        self.assertTrue(c.packets)


    def test_dict_to_cai(self):
        mock_writer = MockWriter()
        counter.dict_to_cai({'a': self.c_list}, mock_writer)
        self.assertEqual(mock_writer.data, '''test +
test + +
test + +
test + +
test + +
test + + +
test + + + +
''')
#         self.assertEqual(tw.data, '''test 600
# test 600 600
# test 600 600
# test 600 600
# test 600 600
# test 600 600 600
# test 600 600 600 600
# ''')

    def test_dict_to_panchenko(self):
        # test creation of panchenko dir, then back, check that the same
        try:
            testdir = tempfile.mkdtemp(dir="/run/user/{}".format(os.geteuid()))
        except OSError:
            testdir = tempfile.mkdtemp()
        counter.dict_to_panchenko({'a': self.c_list}, testdir)
        restored = counter.all_from_panchenko(testdir + '/output-tcp')
        self.assertEqual(
            restored['a'], self.c_list,
            'unlike\n{}\n\n{}'.format([str(x) for x in restored['a']],
                                      [str(x) for x in self.c_list]))


    def test_outlier_removal(self):
        # take simple dict which removes stuff
        # by methods 1-3 (one each)
        # check that original is still the same
        c_dict = {'url': self.c_list}
        c_dict['url'].extend(self.big_val)
        self.assertEqual(len(counter.outlier_removal(c_dict, 1)['url']), 7)
        self.assertEqual(len(c_dict['url']), 8, 'has side effect')
        self.assertEqual(len(counter.outlier_removal(c_dict, 1)['url']), 7)

    def test_p_or_tiny(self):
        with_0 = self.c_list[:]
        with_0.append(counter._test(0))
        fixed = self.c_list[0]
        self.assertEqual(len(counter.p_or_tiny(self.c_list)), 6)
        self.assertEqual(len(counter.p_or_tiny(with_0)), 6)
        self.assertEqual(len(with_0), 8, 'has side effect')
        self.assertEqual(len(self.c_list), 7, 'has side effect')
        self.assertEqual(self.c_list[0], fixed, 'has side effect')

    def test_p_or_toolong(self):
        too_long = [counter._test(3, millisecs=4*60*1000)]
        self.assertEqual(len(counter.p_or_toolong(self.c_list)), 7)
        self.assertEqual(len(self.c_list), 7, 'has side effect')
        self.assertEqual(len(counter.p_or_toolong(too_long)), 0)
        self.assertEqual(len(too_long), 1, 'has side effect')
        self.assertEqual(len(counter.p_or_toolong(self.big_val)), 1)
        self.assertEqual(len(self.big_val), 1, 'has side effect')


    def test_tf_cumul_background(self):
        X, y, yd = counter.to_features_cumul({'background': self.c_list})
        self.assertTrue(-1 in y, '-1 not in {}'.format(set(y)))

    def test_tf_cumul_foreground(self):
        X, y, yd = counter.to_features_cumul({'a': self.c_list})
        self.assertFalse(-1 in y)

    def test_tf_herrmann(self):
        X, y, yd = counter.to_features_herrmann({'url': self.c_list[:2]})
        self.assertEquals(list(X.flatten()), [1, 1])
        self.assertEquals(list(y.flatten()), [0, 0])
        self.assertEquals(yd, ['url', 'url'])


class TestAnalyse(unittest.TestCase):
    '''tests the analyse module'''

    def setUp(self):
        self.c_list = [counter._test(x) for x in [1, 2, 2, 2, 2, 3, 4]]
        self.bg_mock = {'background': self.c_list[:],
                                 'a': self.c_list[:],
                                 'b': self.c_list[:]}

    # todo: use doctest.DocFileSuite
    def test_doc(self):
        (fail_num, _) = doctest.testmod(analyse)
        self.assertEqual(0, fail_num)


    def test__binarize(self):
        res = analyse._binarize(self.bg_mock)
        self.assertEquals(res['background'], self.c_list)
#        self.assertEquals(res['foreground'], 2 * self.c_list)
        self.assertEquals(len(res['foreground']), 2 * len(self.c_list))

    def test__binarize_tf_cumul(self):
        Xa, ya, _ = counter.to_features_cumul(analyse._binarize(self.bg_mock))
        Xc, yc, _ = counter.to_features_cumul(self.bg_mock)
        yc = fit._lb(yc)
        self.assertTrue(np.array_equal(ya, yc))
        self.assertTrue(np.array_equal(Xa, Xc))


class TestFit(unittest.TestCase):
    '''tests the fit module'''

    def setUp(self):
        #self.size = 100
        self.size = 30
        self.X = [(1, 0)] * self.size; self.X.extend([(0, 1)] * self.size)
        self.y = [1] * self.size; self.y.extend([-1] * self.size)
        self.string = 'tpr: {}, fpr: {}'
        fit.FOLDS = 2
        reload(logging)
        logging.basicConfig(level=logging.ERROR) # reduce fit verbosity


    def test_doc(self):
        (fail_num, _) = doctest.testmod(fit)
        self.assertEqual(0, fail_num)


    def test_ow(self):
        '''tests normal open world grid search'''
        result = fit.my_grid(self.X, self.y, auc_bound=0.3)
        self.assertAlmostEqual(result.best_score_, 0.3)

    def test_ow_roc(self):
        '''tests roc for normal open world grid search'''
        (clf, _, _) = fit.my_grid(self.X, self.y, auc_bound=0.3)
        (fpr, tpr, _, _) = fit.roc(clf, self.X, self.y, self.X, self.y)
        self.assertEqual(list(fpr)[:2], [0, 1])
        self.assertEqual(list(tpr)[:2], [1, 1])

    def test_ow_random_minus(self):
        '''tests some class bleed-off: some negatives with same
        feature as positives'''
        X_rand_middle = [(0.5, 0.5)] * (11 * self.size / 10)
        X_rand_middle.extend(np.random.random_sample((9 * self.size / 10, 2)))
        (clf, _, _) = fit.my_grid(X_rand_middle, self.y, auc_bound=0.3)
        (fpr, tpr, _, _) = fit.roc(clf, X_rand_middle, self.y,
                                   X_rand_middle, self.y)
        # 2. check that fpr/tpr has certain structure (low up to tpr of 0.1))
        self.assertEqual(tpr[0], 0, self.string.format(tpr, fpr))
        self.assertEqual(fpr[0], 0, self.string.format(tpr, fpr))
        self.assertEqual(tpr[1], 1, self.string.format(tpr, fpr))
        self.assertEqual(fpr[1], 0.1, self.string.format(tpr, fpr))

    def test_ow_random_plus(self):
        '''tests some class bleed-off: some positives with same
        feature as negatives'''
        X_rand_middle = [(0.5, 0.5)] * (9 * self.size / 10)
        X_rand_middle.extend(np.random.random_sample((11 * self.size / 10, 2)))
        (clf, _, _) = fit.my_grid(X_rand_middle, self.y, auc_bound=0.3)
        (fpr, tpr, _, _) = fit.roc(
            clf, X_rand_middle, self.y, X_rand_middle, self.y)
        # 2. check that fpr/tpr has good structure (rises straight up to 0.9fpr)
        self.assertEqual(tpr[0], 0.9, self.string.format(tpr, fpr))
        self.assertEqual(fpr[0], 0, self.string.format(tpr, fpr))


class TestScenario(unittest.TestCase):
    def setUp(self):
        self.base_mock = {'a': (10, -1), 'b': (10, -1)}
        self.base_mock2 = {'a': (10, -1), 'b': (10, -1), 'c': (10, -1)}


    def test_doc(self):
        (fail_num, _) = doctest.testmod(scenario)
        self.assertEqual(0, fail_num)


    def test__parse_id(self):
        self.assertEqual('disabled',
                         str(scenario.Scenario('disabled/2016-11-13')))
        self.assertEqual(datetime.date(2016, 5, 12),
                         scenario.Scenario('disabled/05-12@10').date)
        self.assertEqual(datetime.date(2016, 7, 6),
                         scenario.Scenario('disabled/bridge--2016-07-06').date)
        self.assertEqual(datetime.date(2016, 11, 4),
                         scenario.Scenario('./0.22/10aI--2016-11-04-50-of-100')
                         .date)
        self.assertEqual('0.22',
                         scenario.Scenario('./0.22/10aI--2016-11-04-50-of-100')
                         .name)
        self.assertEqual(datetime.date(2016, 7, 5),
                         scenario.Scenario('wtf-pad/bridge--2016-07-05').date)
        self.assertEqual('wtf-pad',
                         scenario.Scenario('wtf-pad/bridge--2016-07-05').name)
        self.assertEqual('retro',
                         scenario.Scenario('retro/1').name)
        self.assertEqual('1',
                         scenario.Scenario('retro/1').setting)
        self.assertEqual('0.21',
                         scenario.Scenario('0.21').name)
        self.assertEqual('0.15.3',
                         scenario.Scenario('0.15.3/json-10-nocache').name)
        self.assertEqual('json-10-nocache',
                         scenario.Scenario('0.15.3/json-10-nocache').setting)
        self.assertEqual(
            'bridge', scenario.Scenario('disabled/bridge--2016-07-06').setting)
        #'disabled/nobridge--2016-12-26-with7777' # what to do?


    def test_date_from_trace(self):
        trace = counter._test(3)
        trace.name = u'./msn.com@1467754328'
        s = scenario.Scenario('wtf-pad/bridge--2016-07-05')
        s.traces = {'msn.com' : [trace]}
        self.assertEqual(datetime.date(2016, 7, 5), s.date_from_trace())
        

    def test__filter_all(self):
        self.assertTrue('disabled/2016-06-30' in scenario._filter_all(
            ['disabled', 'disabled/2016-06-30']))


    def test_size_increase__disabled(self):
        self.assertEqual(
            0, scenario.Scenario('disabled/05-12@10').size_increase())


    def test_size_increase__empty(self):
        trace = counter._test(0)
        s = scenario.Scenario('wtf-pad/2015-01-01')
        s.traces = {'msn.com': [trace], 'apple.com': [trace]}
        self.assertEqual(-100, s.size_increase())


    def test__size_increase_computation_equal(self):
        self.assertEqual(scenario._size_increase_computation(self.base_mock,
                                                {'a': (10, -1), 'b': (10, -1)}),
                         0)

    def test__size_increase_computation_same_half(self):
        self.assertEqual(scenario._size_increase_computation(self.base_mock,
                                                {'a': (5, -1), 'b': (5, -1)}),
                         -50)

    def test__size_increase_computation_same_double(self):
        self.assertEqual(scenario._size_increase_computation(self.base_mock,
                                                {'a': (20, -1), 'b': (20, -1)}),
                         100)

    def test__size_increase_computation_one_double(self):
        self.assertAlmostEqual(
            scenario._size_increase_computation(self.base_mock,
                                   {'a': (10, -1), 'b': (20, -1)}),
            50)
#                               100*(pow(2, 1./2) - 1))#harmonic

    def test__size_increase_computation_both_different(self):
        self.assertEqual(scenario._size_increase_computation(self.base_mock,
                                                {'a': (5, -1), 'b': (20, -1)}),
                         25)
#                         0)# harmonic

    def test__size_increase_computation_three_one(self):
        self.assertAlmostEqual(scenario._size_increase_computation(
            self.base_mock2, {'a': (10, -1), 'b': (10, -1), 'c': (20, -1)}),
                               100/3.)
#                               100.*(pow(2, 1./3)-1))#harmonic


    def test__size_increase_computation_three_one_reverted(self):
        self.assertAlmostEqual(scenario._size_increase_computation(
            {'a': (10, -1), 'b': (10, -1), 'c': (20, -1)}, self.base_mock2),
                               250./3-100)
#                               100.*(pow(1./2, 1./3)-1))#harmonic


class MockWriter(object):
    '''simulates file-like object with =write= method'''
    def __init__(self):
        self.data = ''

    def write(self, line):
        self.data += line


if __name__ == '__main__':
    unittest.main()
