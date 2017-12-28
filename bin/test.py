#! /usr/bin/env python
'''unit tests counter, analyse and fix modules'''
import datetime
import doctest
import logging
import os
import subprocess
import tempfile
import time
import unittest

import numpy as np
from unittest.runner import TextTestRunner, TextTestResult

import analyse
import config
import counter
import fit
import scenario
import results
from capture import one_site

config.FOLDS = 2


VERYQUICK = os.getenv('VERYQUICK', False)
QUICK = os.getenv('QUICK', False) or VERYQUICK


class TestCounter(unittest.TestCase):
    '''tests the counter module'''
    def setUp(self):
        self.c_list = [counter._test(x) for x in [1, 2, 2, 2, 2, 3, 4]] # len 7
        self.big_val = [counter._test(3, val=10*60*1000)] # very big, but ok
        logging.disable(logging.INFO) # change to .INFO or disable for debug

    def tearDown(self):
        logging.disable(logging.NOTSET)

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
        testdir = temp_dir()
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

    def test__from(self):
        current = os.getcwd()
        emptydir = temp_dir()
        os.chdir(emptydir)
        with self.assertRaises(IOError):
            counter.Counter.from_('path/to/counter.py',)
        os.chdir(current)


class TestExp(unittest.TestCase):
    '''tests the experimentation module'''
    @unittest.skipIf(VERYQUICK, "slow test skipped")
    def test_runs(self):
#        subprocess.call("echo $PATH > /tmp/out",
        with open(os.devnull, "w") as null:
            subprocess.check_call(
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "./exp.py")
                + " print_config > /dev/null",
                stderr=null, stdout=null, shell=True)


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


class TestFit(unittest.TestCase):
    '''tests the fit module'''

    def setUp(self):
        self.size = 100
        self.X, self.y = _init_X_y(self.size)
        self.string = 'tpr: {}, fpr: {}'
        fit.FOLDS = 2
        logging.disable(logging.WARNING) # change to .INFO or disable for debug

    def tearDown(self):
        logging.disable(logging.NOTSET)

    @unittest.skipIf(VERYQUICK, "slow test skipped")
    def test_doc(self):
        '''runs doctests'''
        (fail_num, _) = doctest.testmod(fit)
        self.assertEqual(0, fail_num)

    @unittest.skipIf(QUICK, "slow test skipped")
    def test_ow(self):
        '''tests normal open world grid search'''
        result = fit.my_grid(self.X, self.y, auc_bound=0.3)
        self.assertAlmostEqual(result.best_score_, 1)

    @unittest.skipIf(QUICK, "slow test skipped")
    def test_ow_roc(self):
        '''tests roc for normal open world grid search'''
        (self.X, self.y) = _init_X_y(100, False)
        (clf, _, _) = fit.my_grid(self.X, self.y, auc_bound=0.3)
        (fpr, tpr, _, _) = fit.roc(clf, self.X, self.y, self.X, self.y)
        self.assertEqual(list(fpr)[:2], [0, 1])
        self.assertEqual(list(tpr)[:2], [1, 1])

    @unittest.skipIf(QUICK, "slow test skipped")
    def test_ow_minus(self):
        '''tests some class bleed-off: some negatives with same
        feature as positives'''
        self.X, self.y = _init_X_y(self.size)
        X_rand_middle = [(1, 0)] * (11 * self.size / 10)
        #X_rand_middle.extend(np.random.random_sample((9 * self.size / 10, 2)))
        X_rand_middle.extend([(0, 1)] * (9 * self.size / 10))
        (clf, _, _) = fit.my_grid(X_rand_middle, self.y, auc_bound=0.3)
        (fpr, tpr, _, _) = fit.roc(clf, X_rand_middle, self.y,
                                   X_rand_middle, self.y)
        # 2. check that fpr/tpr has certain structure (low up to tpr of 0.1))
        self.assertEqual(tpr[0], 0, self.string.format(tpr, fpr))
        self.assertEqual(fpr[0], 0, self.string.format(tpr, fpr))
        self.assertEqual(tpr[1], 1, self.string.format(tpr, fpr))
        self.assertEqual(fpr[1], 0.1, self.string.format(tpr, fpr))

    @unittest.skipIf(QUICK, "slow test skipped")
    def test_ow_random_plus(self):
        '''tests some class bleed-off: some positives with same
        feature as negatives'''
        X, y = _init_X_y(self.size)
        X_rand_middle = [(0.5, 0.5)] * (9 * self.size / 10)
        X_rand_middle.extend(np.random.random_sample((11 * self.size / 10, 2)))
        (clf, _, _) = fit.my_grid(X_rand_middle, self.y, auc_bound=0.3)
        (fpr, tpr, _, _) = fit.roc(
            clf, X_rand_middle, self.y, X_rand_middle, self.y)
        # 2. check that fpr/tpr has good structure (rises straight up to 0.9fpr)
        self.assertEqual(tpr[0], 0.9, self.string.format(tpr, fpr))
        self.assertEqual(fpr[0], 0, self.string.format(tpr, fpr))


class TestResult(unittest.TestCase):
    '''tests the counter module'''

    @unittest.skipIf(QUICK, "slow test skipped")
    def test_doc(self):
        (fail_num, _) = doctest.testmod(results, optionflags=doctest.ELLIPSIS)
        self.assertEqual(0, fail_num)


class TestScenario(unittest.TestCase):
    def setUp(self):
        self.base_mock = {'a': (10, -1), 'b': (10, -1)}
        self.base_mock2 = {'a': (10, -1), 'b': (10, -1), 'c': (10, -1)}
        logging.disable(logging.WARNING)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_doc(self):
        (fail_num, _) = doctest.testmod(scenario)
        self.assertEqual(0, fail_num)

    def test___init__(self):
        self.assertEqual('no defense',
                         str(scenario.Scenario('disabled/2016-11-13')))
        self.assertEqual(datetime.date(2016, 5, 12),
                         scenario.Scenario('disabled/05-12@10').date)
        self.assertEqual(datetime.date(2016, 7, 6),
                         scenario.Scenario('disabled/bridge--2016-07-06').date)
        self.assertEqual(datetime.date(2016, 11, 4),
                         scenario.Scenario('./0.22/10aI--2016-11-04-50-of-100')
                         .date)
        self.assertEqual('new defense',
                         scenario.Scenario('./0.22/10aI--2016-11-04-50-of-100')
                         .name)
        self.assertEqual(datetime.date(2016, 7, 5),
                         scenario.Scenario('wtf-pad/bridge--2016-07-05').date)
        self.assertEqual('wtf-pad',
                         scenario.Scenario('wtf-pad/bridge--2016-07-05').name)
        self.assertEqual(
            '5', scenario.Scenario(u'simple2/5--2016-09-23-100').setting)
        # retro traces are gone: were worse than all others
        # self.assertEqual('retro',
        #                  scenario.Scenario('retro/1').name)
        # self.assertEqual('1',
        #                  scenario.Scenario('retro/1').setting)
        self.assertEqual('0.21',
                         scenario.Scenario('0.21').name)
        self.assertEqual('0.15.3',
                         scenario.Scenario('0.15.3/json-10-nocache').name)
        # new name for above
        self.assertEqual('nocache',
                         scenario.Scenario('0.15.3/nocache--2016-06-17--10@30').setting)
        self.assertEqual(
            'bridge', scenario.Scenario('disabled/bridge--2016-07-06').setting)
        self.assertEqual(datetime.date(2017, 9, 6),
                         scenario.Scenario("wtf-pad/bridge--2017-09-06").date)
        #'disabled/nobridge--2016-12-26-with7777' # what to do?

    def test___equal__(self):
        self.assertEqual(scenario.Scenario("wtf-pad/bridge--2017-09-06"),
                         scenario.Scenario("wtf-pad/bridge--2017-09-06"))
        self.assertNotEqual(scenario.Scenario("0.20/0-ai--2016-06-25"),
                            scenario.Scenario("0.20/20-ai--2016-06-25"))
        self.assertNotEqual(scenario.Scenario("0.20/0-ai--2016-06-25"),
                            scenario.Scenario("0.20/0-aii--2016-06-25"))

    def test_binarize_fake(self):
        c_list = [counter._test(x) for x in [1, 2, 2, 2, 2, 3, 4]]
        bg_mock = {'background': c_list[:],
                   'a': c_list[:],
                   'b': c_list[:]}
        s = scenario.Scenario('asdf/12-12@20')
        s.traces = bg_mock
        res = s.binarize().get_traces()
        self.assertEquals(res['background'], c_list)
        self.assertEquals(len(res['foreground']), 2 * len(c_list))

    def test__binarize_fake_vs_fit(self):
        c_list = [counter._test(x) for x in [1, 2, 2, 2, 2, 3, 4]]
        bg_mock = {'background': c_list[:],
                   'a': c_list[:],
                   'b': c_list[:]}
        s = scenario.Scenario('asdf/12-12@20')
        s.traces = bg_mock
        Xa, ya, _ = s.binarize().get_features_cumul()
        Xc, yc, _ = counter.to_features_cumul(bg_mock)
        yc = fit._lb(yc)
        self.assertTrue(np.array_equal(ya, yc), "ya: {}\nyc: {}".format(ya, yc))
        self.assertTrue(np.array_equal(Xa, Xc))

    @unittest.skipIf(QUICK, "slow test skipped")
    def test_binarize_real(self):
        s = scenario.Scenario('wtf-pad/bridge--2016-07-05')
        self.assertTrue(2,
                        len(s.get_open_world().binarize().get_traces().keys()))

    def test_date_from_trace(self):
        trace = counter._test(3)
        trace.name = u'./msn.com@1467754328'
        s = scenario.Scenario('wtf-pad/bridge--2016-07-05')
        s.traces = {'msn.com' : [trace]}
        self.assertEqual(datetime.date(2016, 7, 5), s.date_from_trace())

    @unittest.skipIf(QUICK, "slow test skipped")
    def test_get_open_world(self):
        s = scenario.Scenario('disabled/05-12@10')
        self.assertTrue('background' in s.get_open_world().get_traces())

    def test_sample(self):
        s = scenario.Scenario('somescenario/2012-07-05')
        s.traces = {'msn.com' : [counter._test(3)] * 30}
        self.assertEqual(len(s.get_sample(10)['msn.com']), 10)

    def test_size_increase__disabled(self):
        self.assertEqual(
            0, scenario.Scenario('disabled/05-12@10').size_increase())

    @unittest.skipIf(QUICK, "slow test skipped")
    def test_size_increase__empty(self):
        trace = counter._test(0)
        s = scenario.Scenario('wtf-pad/bridge--2016-07-05')
        s.traces = {'msn.com': [trace], 'google.com': [trace]}
        self.assertEqual(-100, s.size_increase())

    def test__filter_all(self):
        self.assertTrue('disabled/2016-06-30' in scenario._filter_all(
            ['disabled', 'disabled/2016-06-30'], True))
        self.assertTrue('disabled/2016-06-30' in scenario._filter_all(
            ['disabled', 'disabled/2016-06-30'], False))

    @unittest.skipIf(QUICK, "slow test skipped")
    def test__closest_bg(self):
        s = scenario.Scenario('disabled/background--2016-08-17')
        self.assertEqual(s, s._closest('background', include_bg=True))

    def test__compute_increase_equal(self):
        self.assertEqual(
            scenario._compute_increase(self.base_mock,
                                       {'a': (10, -1), 'b': (10, -1)}),
            0)

    def test__compute_increase_same_half(self):
        self.assertEqual(scenario._compute_increase(
            self.base_mock, {'a': (5, -1), 'b': (5, -1)}),
                         -50)

    def test__compute_increase_same_double(self):
        self.assertEqual(scenario._compute_increase(
            self.base_mock, {'a': (20, -1), 'b': (20, -1)}),
                         100)

    def test__compute_increase_one_double(self):
        self.assertAlmostEqual(scenario._compute_increase(
            self.base_mock, {'a': (10, -1), 'b': (20, -1)}),
                               50)
#                               100*(pow(2, 1./2) - 1))#harmonic

    def test__compute_increase_both_different(self):
        self.assertEqual(scenario._compute_increase(
            self.base_mock, {'a': (5, -1), 'b': (20, -1)}),
                         25)
#                         0)# harmonic

    def test__compute_increase_three_one(self):
        self.assertAlmostEqual(scenario._compute_increase(
            self.base_mock2, {'a': (10, -1), 'b': (10, -1), 'c': (20, -1)}),
                               100/3.)
#                               100.*(pow(2, 1./3)-1))#harmonic


    def test__compute_increase_three_one_reverted(self):
        self.assertAlmostEqual(scenario._compute_increase(
            {'a': (10, -1), 'b': (10, -1), 'c': (20, -1)}, self.base_mock2),
                               250./3-100)
#                               100.*(pow(1./2, 1./3)-1))#harmonic


class TestCaptureOnesite(unittest.TestCase):
    '''tests the one_site module/program'''
    def setUp(self):
        logging.disable(logging.WARNING) # change to .INFO or disable for debug

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_doc(self):
        (fail_num, _) = doctest.testmod(counter, optionflags=doctest.ELLIPSIS)
        self.assertEqual(0, fail_num)

    def test__check_text(self):
        with self.assertRaises(one_site.DelayError):
            one_site._check_text("test tor network settings")
        # typeerrors: trying to rename None file
        with self.assertRaises(TypeError):
            one_site._check_text("hello")
        one_site._check_text("hello\n\n\n\n")
        with self.assertRaises(TypeError):
            one_site._check_text("Reference #18\n\n\n\n")


class MockWriter(object):
    '''simulates file-like object with =write= method'''
    def __init__(self):
        self.data = ''

    def write(self, line):
        self.data += line


class TimeLoggingTestResult(TextTestResult):
    SLOW_TEST_THRESHOLD = 0.3

    def startTest(self, test):
        self._started_at = time.time()
        super(TimeLoggingTestResult, self).startTest(test)

    def addSuccess(self, test):
        elapsed = time.time() - self._started_at
        if elapsed > self.SLOW_TEST_THRESHOLD:
            name = self.getDescription(test)
            self.stream.write(
                "\n{} ({:.03}s)\n".format(
                    name, elapsed))
        super(TimeLoggingTestResult, self).addSuccess(test)


def temp_dir():
    try:
        return tempfile.mkdtemp(dir="/run/user/{}".format(os.geteuid()))
    except OSError:
        return tempfile.mkdtemp()


def _init_X_y(size, random=True):
    '''@return (X, y) of given size'''
    X = [(1, 0)] * size;
    X.extend([(0, 1)] * size)
    if random:
        X = np.array(X).astype('float64')
        X += np.random.random_sample(X.shape) * 0.8 -0.4
    y = [1] * size; y.extend([-1] * size)
    return (X, y)


if __name__ == '__main__':
    test_runner = TextTestRunner(resultclass=TimeLoggingTestResult)
    unittest.main(testRunner=test_runner)
