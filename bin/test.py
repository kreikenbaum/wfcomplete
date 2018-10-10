#! /usr/bin/env python
'''unit tests counter, analyse and fix modules'''
import datetime
import doctest
import json
import logging
import os
import socket
import subprocess
import tempfile
import time
import unittest
from unittest.runner import TextTestRunner, TextTestResult

import numpy as np
from sklearn import metrics

import analyse
import config
import counter
import fit
import mymetrics
import scenario
import results
from capture import one_site, utils


config.FOLDS = 2


VERYQUICK = os.getenv('VERYQUICK', False)
QUICK = os.getenv('QUICK', False) or VERYQUICK
ALWAYS = True


class TestAnalyse(unittest.TestCase):
    '''tests the analyse module'''

    def setUp(self):
        self.c_list = [counter._test(x) for x in [1, 2, 2, 2, 2, 3, 4]]
        self.bg_mock = {'background': self.c_list[:],
                        'a': self.c_list[:],
                        'b': self.c_list[:]}

    # todo: use doctest.DocFileSuite
    def test_doc(self):
        '''test all analyse's docstrings'''
        (fail_num, _) = doctest.testmod(analyse)
        self.assertEqual(0, fail_num)


class TestCaptureOnesite(unittest.TestCase):
    '''tests the one_site module/program'''
    def setUp(self):
        logging.disable(logging.WARNING)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_doc(self):
        '''test all one_site's docstrings'''
        (fail_num, _) = doctest.testmod(counter, optionflags=doctest.ELLIPSIS)
        self.assertEqual(0, fail_num)

    def test__check_text(self):
        '''test website text check function'''
        with self.assertRaises(one_site.DelayError):
            one_site._check_text("test tor network settings")
        # typeerrors: trying to rename None file
        with self.assertRaises(TypeError):
            one_site._check_text("hello")
        one_site._check_text("hello\n\n\n\n")
        with self.assertRaises(TypeError):
            one_site._check_text("Reference #18\n\n\n\n")


class TestCaptureUtils(unittest.TestCase):
    '''tests the utils module'''

    def test_site(self):
        '''call site(), get results'''
        self.assertEqual("mlsec", utils.site({"host": "duckstein"}))
        self.assertEqual(
            "gcloud",
            utils.site({"host": "main-test.c.pioneering-mode-193216.internal"})
        )


class TestConfig(unittest.TestCase):
    '''tests the config module'''
    def test_matches_addon(self):
        '''test that addon preset matches'''
        try:
            testdir = os.path.dirname(os.path.abspath(__file__))
            target = os.path.join(testdir, '..', 'cover', 'package.json')
            with open(target) as f:
                package = json.load(f)
                for pref in package['preferences']:
                    if pref['name'] == 'Traffic-HOST':
                        self.assertEqual(config.MAIN, pref['value'])
        except IOError:
            logging.info("cover addon not found at ../cover: %s", target)


class TestCounter(unittest.TestCase):
    '''tests the counter module'''
    def setUp(self):
        self.c_list = [counter._test(x) for x in [1, 2, 2, 2, 3, 3, 4]]  # len7
        self.big_val = [counter._test(3, val=10*60*1000)]  # very big, but ok
        logging.disable(logging.INFO)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_doc(self):
        '''test counter doctests'''
        (fail_num, _) = doctest.testmod(counter, optionflags=doctest.ELLIPSIS)
        self.assertEqual(0, fail_num)

    def test__test(self):
        '''tests dummy method'''
        trace = counter._test(35)
        self.assertTrue(trace.timing)
        self.assertTrue(trace.packets)

    def test_dict_to_cai(self):
        '''test to cai conversion'''
        mock_writer = MockWriter()
        counter.dict_to_cai({'a': self.c_list}, mock_writer)
        # this corresponds to c_list, if that is changed, change this, too
        self.assertEqual(mock_writer.data, '''test +
test + +
test + +
test + +
test + + +
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
        '''test creation of panchenko dir, then back, check that the same'''
        testdir = temp_dir()
        counter.dict_to_panchenko({'a': self.c_list}, testdir)
        restored = counter.all_from_panchenko(testdir + '/output-tcp')
        self.assertEqual(
            restored['a'], self.c_list,
            'unlike\n{}\n\n{}'.format([str(x) for x in restored['a']],
                                      [str(x) for x in self.c_list]))

    def test_outlier_removal(self):
        '''take simple dict which removes stuff by method 1 (one each),
        check that original is still the same'''
        c_dict = {'url': self.c_list}
        c_dict['url'].extend(self.big_val)
        # tiny is always removed
        self.assertEqual(len(counter.outlier_removal(c_dict, 0)['url']), 7)
        self.assertEqual(len(counter.outlier_removal(c_dict, 1)['url']), 7)
        self.assertEqual(len(c_dict['url']), 8, 'has side effect')

    def test_p_or_tiny(self):
        '''test tiny outlier removal: 1 of c_list is removed'''
        with_0 = self.c_list[:]
        with_0.append(counter._test(0))
        fixed = self.c_list[0]
        self.assertEqual(len(counter.p_or_tiny(self.c_list)), 6)
        self.assertEqual(len(counter.p_or_tiny(with_0)), 6)
        self.assertEqual(len(with_0), 8, 'has side effect')
        self.assertEqual(len(self.c_list), 7, 'has side effect')
        self.assertEqual(self.c_list[0], fixed, 'has side effect')

    def test_p_or_toolong(self):
        '''test outlier removal of too-long trace'''
        too_long = [counter._test(3, millisecs=config.DURATION_LIMIT*1000)]
        self.assertEqual(len(counter.p_or_toolong(self.c_list)), 7)
        self.assertEqual(len(self.c_list), 7, 'has side effect')
        self.assertEqual(len(counter.p_or_toolong(too_long)), 0)
        self.assertEqual(len(too_long), 1, 'has side effect')
        self.assertEqual(len(counter.p_or_toolong(self.big_val)), 1)
        self.assertEqual(len(self.big_val), 1, 'has side effect')
        c_dict = {'url': list(self.c_list)}
        c_dict['url'].pop(0)
        # no change if all ok
        self.assertEqual(len(counter.outlier_removal(c_dict, 0, True)['url']), 6)
        # too long is removed
        self.assertTrue(
            'url' not in counter.outlier_removal({"url": too_long}, 0, True))


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
        with open(os.devnull, "w") as null:
            subprocess.check_call(
                os.path.join(utils.path_to(__file__), "./exp.py")
                + " print_config > /dev/null",
                stderr=null, stdout=null, shell=True)


class TestExpOpen(unittest.TestCase):
    '''tests the open-world experimentation module'''
    @unittest.skipIf(VERYQUICK, "slow test skipped")
    def test_runs(self):
        with open(os.devnull, "w") as null:
            subprocess.check_call(
                os.path.join(utils.path_to(__file__), "./exp_open_world.py")
                + " print_config > /dev/null",
                stderr=null, stdout=null, shell=True)


class TestFit(unittest.TestCase):
    '''tests the fit module'''

    def setUp(self):
        self.size = 100
        self.X, self.y = _init_X_y(self.size)
        self.string = 'tpr: {}, fpr: {}'
        fit.FOLDS = 2
        logging.disable(logging.WARNING)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    @unittest.skipIf(VERYQUICK, "slow test skipped")
    def test_doc(self):
        '''runs doctests'''
        (fail_num, _) = doctest.testmod(fit)
        self.assertEqual(0, fail_num)

    @unittest.skipIf(QUICK, "slow test skipped")
    def test_open_world(self):
        '''tests normal open world grid search'''
        result = fit.my_grid(self.X, self.y, auc_bound=0.3)
        self.assertAlmostEqual(result.best_score_, 1)

    @unittest.skipIf(QUICK, "slow test skipped")
    def test_open_world_roc(self):
        '''tests roc for normal open world grid search'''
        clf, _, _ = fit.my_grid(self.X, self.y, auc_bound=0.3)
        fpr, tpr, _, _ = fit.roc(clf, self.X, self.y)
        self.assertEqual(zip(fpr, tpr)[-2:], [(0.0, 1.0), (1.0, 1.0)])

    @unittest.skipIf(QUICK, "slow test skipped")
    def test_open_world_minus(self):
        '''tests some class bleed-off: some negatives with same
        feature as positives'''
        X_middle = [(1, 0)] * (11 * self.size / 10)
        # X_middle.extend(np.random.random_sample((9 * self.size / 10, 2)))
        X_middle.extend([(0, 1)] * (9 * self.size / 10))
        (clf, _, _) = fit.my_grid(X_middle, self.y, auc_bound=0.3)
        (fpr, tpr, _, _) = fit.roc(clf, X_middle, self.y)
        # 2. check that fpr/tpr has certain structure (low up to tpr of 0.1))
        self.assertEqual(tpr[0], 0, self.string.format(tpr, fpr))
        self.assertEqual(fpr[0], 0, self.string.format(tpr, fpr))
        self.assertTrue((0.1, 1) in zip(fpr, tpr),
                        self.string.format(tpr, fpr))

    @unittest.skipIf(QUICK, "slow test skipped")
    def test_open_world_random_plus(self):
        '''tests some class bleed-off: some positives with same
        feature as negatives'''
        self.X = [(1.01, 1.01)] * (9 * self.size / 10)
        self.X.extend(np.random.random_sample((11 * self.size / 10, 2)))
        clf, _, _ = fit.my_grid(self.X, self.y, auc_bound=0.01)
        fpr, tpr, _, _ = fit.roc(clf, self.X, self.y)
        # 2. check that fpr/tpr has good structure (rises directly to 0.9fpr)
        self.assertTrue((0.9, 0.0) in zip(tpr, fpr),
                        msg='{}\n{}'.format(zip(tpr, fpr), clf))
        #self.assertAlmostEqual(tpr[1], 0.9,
        #                       msg='{}\n{}'.format(zip(tpr, fpr), clf))
        #self.assertAlmostEqual(fpr[1], 0, zip(tpr, fpr))


class TestMymetrics(unittest.TestCase):
    '''tests the counter module'''
    def setUp(self):
        self.size = 100
        self.X, self.y = _init_X_y(self.size)
        logging.disable(logging.WARNING)  # change to .INFO / disable for debug
        self.old_jobs = config.JOBS_NUM
        config.JOBS_NUM = 1
        # from sklearn metrics doc string
        y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
        y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
        self.cm = metrics.confusion_matrix(y_true, y_pred,
                                           labels=["ant", "bird", "cat"])

    def tearDown(self):
        logging.disable(logging.NOTSET)
        config.JOBS_NUM = self.old_jobs

    def test_doc(self):
        (fail_num, _) = doctest.testmod(mymetrics,
                                        optionflags=doctest.ELLIPSIS)
        self.assertEqual(0, fail_num)

    def test_binarize_probability(self):
        self.assertTrue(np.all(
            mymetrics.binarize_probability(
                np.array([[0.5, 0.2, 0.3], [0.3, 0.2, 0.5]]))
            == np.array([[0.5, 0.5], [0.3, 0.7]])))

    def test_tprfpr(self):
        '''test all tpr-fpr extraction'''
        right_cm = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual([(1, 0, 1), (1, 0, 1), (1, 0, 1)],
                         mymetrics.tpr_fpr_tpa(right_cm))
#        wrong_cm = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
#        self.assertEqual([(0, 1), (0, 1), (0, 1)],
#                         analyse.tpr_fpr_tpa(wrong_cm))
        # ant: 2 tp (ant pred. as ant), 0 fn (ant as sth else),
        # 1 fp  (cat pred. as ant), 3 tn (cat/bird pred. as cat)
        # \to rp (tp+fn) = 2, rn (fp+tn) = 4, pp (tp+fp) = 3, rp (tp+fn) = 2
        # cat: 2 tp, 1 fn, 1 fp, 2 tn \to 3 rp, 3 rn, 3 pp, ...
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', r'invalid value ')
            self.assertEqual(
                [(1.0, 0.25), (0.0, 0.0), (2./3, 1./3)],
                [(t, f) for (t, f, _) in mymetrics.tpr_fpr_tpa(self.cm)])

    def test_tpa(self):
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', r'invalid value ')
            self.assertEqual((1.0, 0.25, 2./3),
                             mymetrics.tpr_fpr_tpa(self.cm)[0])

    # def test_bounded_auc(self):
    #     clf = fit.clf_default(probability=True)
    #     for bnd in [0.01, 0.1, 0.2, 0.5, 1]:
    #         s = metrics.make_scorer(
    #             mymetrics.bounded_auc, needs_proba=True, y_bound=bnd)
    #     dfl = mymetrics.compute_bounded_auc_score(clf, self.X, self.y, bnd)
    #     self.assertEqual(
    #         dfl,
    #         mymetrics.compute_bounded_auc_score(clf, self.X, self.y, bnd, s),
    #         "bound: {}".format(bnd))


class TestResults(unittest.TestCase):
    '''tests the results module'''

    @unittest.skipIf(QUICK, "slow test skipped")
    def test_doc(self):
        (fail_num, _) = doctest.testmod(results, optionflags=doctest.ELLIPSIS)
        self.assertEqual(0, fail_num)

    @unittest.skipIf(QUICK, "slow test skipped")
    def test_open_world_config(self):
        '''
        1. create result with open world
        2. check that its scenario has _open_world_config
        '''
        result = [r for r in results.list_all() if r._id == 549][0]
        self.assertTrue(result.scenario._open_world_config)

    @unittest.skipIf(QUICK, "slow test skipped")
    def test_real_open_world_config(self):
        '''
        1. create result with open world
        2. check that traces have background pages
        '''
        smallresult = min((r for r in results.list_all() if r.open_world),
                          key=lambda x: x.size)
        self.assertTrue("background" in smallresult.scenario.get_traces(
            current_sites=False))

    def test___init__sets_all(self):
        PARAM_NUM = 16  # how to get dynamically?
        with self.assertRaises(TypeError):
            results.Result(*range(PARAM_NUM + 1))
        result = results.Result(*range(PARAM_NUM - 1), src={})
        self.assertEqual(
            PARAM_NUM,
            len([x for x in result.__dict__.values() if x is not None]))

    def test_get_confusion_matrix(self):
        cm = results.Result(
            *range(6), ytrue=[0, 1], ypred=[0, 0],
            src={}).get_confusion_matrix()
        self.assertTrue(np.all(cm == np.array([[1, 0], [1, 0]])))


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
        self.assertEqual('no defense on 2016-11-13',
                         str(scenario.Scenario('disabled/2016-11-13')))
        self.assertEqual(datetime.date(2016, 5, 12),
                         scenario.Scenario("disabled/2016-05-12--10@40").date)
        self.assertEqual(datetime.date(2016, 7, 6),
                         scenario.Scenario('disabled/bridge--2016-07-06').date)
        self.assertEqual(datetime.date(2016, 11, 4),
                         scenario.Scenario('0.22/10aI--2016-11-04--100@50')
                         .date)
        self.assertEqual('SCT',
                         scenario.Scenario('./0.22/10aI--2016-11-04--100@50')
                         .name)
        self.assertEqual(datetime.date(2016, 7, 5),
                         scenario.Scenario('wtf-pad/bridge--2016-07-05').date)
        self.assertEqual('wtf-pad',
                         scenario.Scenario('wtf-pad/bridge--2016-07-05').name)
        self.assertEqual(
            '5', scenario.Scenario(u'simple2/5--2016-09-23--100').setting)
        self.assertEqual('SCT',
                         scenario.Scenario('0.21/2016-06-30').name)
        self.assertEqual(
            '0.21', scenario.Scenario('0.21/2016-06-30').version)
        self.assertEqual(
            'nocache',
            scenario.Scenario('0.15.3/nocache--2016-06-17--10@30').setting)
        self.assertEqual(
            'bridge', scenario.Scenario('disabled/bridge--2016-07-06').setting)
        self.assertEqual(datetime.date(2017, 9, 6),
                         scenario.Scenario("wtf-pad/bridge--2017-09-06").date)
        # 'disabled/nobridge--2016-12-26-with7777' # what to do?

    def test___equal__(self):
        self.assertEqual(scenario.Scenario("wtf-pad/bridge--2017-09-06"),
                         scenario.Scenario("wtf-pad/bridge--2017-09-06"))
        self.assertNotEqual(scenario.Scenario("0.20/0-ai--2016-06-25"),
                            scenario.Scenario("0.20/20-ai--2016-06-25"))
        self.assertNotEqual(scenario.Scenario("0.20/0-ai--2016-06-25"),
                            scenario.Scenario("0.20/0-aii--2016-06-25"))

    def test_background(self):
        self.assertFalse(scenario.Scenario('disabled/2016-11-13').background)
        s = scenario.Scenario('disabled/background--2016-08-17--4100@1')
        self.assertTrue(s.background)

    @unittest.skipIf(QUICK, "slow test skipped")
    def test_open_world_config(self):
        s = scenario.Scenario("disabled/2016-05-12--10@40")
        s._open_world_config = {'binary': False, 'exclude_sites': [],
                                'background_size': None}
        self.assertTrue("background" in s.get_traces().keys())

    @unittest.skipIf(QUICK, "slow test skipped")
    def test_open_world_config_current_sites_true(self):
        s = scenario.Scenario("disabled/2016-05-12--10@40")
        s._open_world_config = {'binary': False, 'exclude_sites': [],
                                'background_size': None, 'current_sites': True}
        self.assertFalse("baidu.com" in s.get_traces().keys())

    @unittest.skipIf(QUICK, "slow test skipped")
    def test_open_world_config_current_sites_false(self):
        s = scenario.Scenario("disabled/2016-05-12--10@40")
        s._open_world_config = {'binary': False, 'exclude_sites': [],
                                'background_size': None,
                                'current_sites': False}
        self.assertTrue("baidu.com" in s.get_traces().keys())

    def test_binarized_fake(self):
        c_list = [counter._test(x) for x in [1, 2, 2, 2, 2, 3, 4]]
        bg_mock = {'background': c_list[:],
                   'a': c_list[:],
                   'b': c_list[:]}
        s = scenario.Scenario('asdf/2015-12-12--3@7')
        s.traces = bg_mock
        res = s.binarized().get_traces()
        self.assertEquals(res['background'], c_list)
        self.assertEquals(len(res['foreground']), 2 * len(c_list))

    def test__binarized_fake_vs_fit(self):
        c_list = [counter._test(x) for x in [1, 2, 2, 2, 2, 3, 4]]
        bg_mock = {'background': c_list[:],
                   'a': c_list[:],
                   'b': c_list[:]}
        s = scenario.Scenario('asdf/2015-12-12--3@7')
        s.traces = bg_mock
        Xa, ya, _ = s.binarized().get_features_cumul(current_sites=False)
        Xc, yc, _ = counter.to_features_cumul(bg_mock)
        yc = list(mymetrics.binarized(yc, transform_to=1))
        self.assertTrue(np.array_equal(ya, yc), "ya:{}\nyc:{}".format(ya, yc))
        self.assertTrue(np.array_equal(Xa, Xc))

    @unittest.skipIf(QUICK, "slow test skipped")
    def test_binarized_real(self):
        s = scenario.Scenario('wtf-pad/bridge--2016-07-05')
        self.assertTrue(
            2, len(s.get_open_world().binarized().get_traces().keys()))

    def test_config(self):
        self.assertEqual(
            '5', scenario.Scenario(u'simple2/5--2016-09-23--100').config)
        self.assertEqual('', scenario.Scenario('0.21/2016-06-30').config)
        self.assertEqual(
            '', scenario.Scenario('0.15.3/nocache--2016-06-17--10@30').config)
        self.assertEqual(
            '10aI',
            scenario.Scenario('0.22/10aI--2016-11-04--100@50').config)

    def test_date_from_trace(self):
        trace = counter._test(3)
        trace.name = u'./msn.com@1467754328'
        s = scenario.Scenario('wtf-pad/bridge--2016-07-05')
        s.traces = {'msn.com': [trace]}
        self.assertEqual(datetime.date(2016, 7, 5), s.date_from_trace())

    @unittest.skipIf(QUICK, "slow test skipped")
    def test_get_open_world(self):
        s = scenario.Scenario("disabled/2016-05-12--10@40")
        self.assertTrue('background' in s.get_open_world().get_traces())

    def test_sample(self):
        s = scenario.Scenario('somescenario/2012-07-05')
        s.traces = {'msn.com': [counter._test(3)] * 30}
        self.assertEqual(len(s.get_sample(10)['msn.com']), 10)

    @unittest.skipIf(QUICK, "slow test skipped")
    def test_size_increase__disabled(self):
        self.assertEqual(
            0, scenario.Scenario("disabled/2016-05-12--10@40").size_increase())

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
        s = scenario.Scenario('disabled/background--2016-08-17--4100@1')
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


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.base_mock = {'a': (10, -1), 'b': (10, -1)}
        self.base_mock2 = {'a': (10, -1), 'b': (10, -1), 'c': (10, -1)}
        logging.disable(logging.WARNING)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_doc(self):
        (fail_num, _) = doctest.testmod(scenario)
        self.assertEqual(0, fail_num)

    def test_clf_default(self):
        clf = utils.clf_default()
        self.assertFalse(clf.estimator.class_weight)
        clf = utils.clf_default([-1, 1])
        self.assertFalse(clf.estimator.class_weight)
        clf = utils.clf_default([0, 1])
        self.assertTrue(clf.estimator.class_weight)


class TestStatus(unittest.TestCase):
    @unittest.skipIf(ALWAYS, "takes veeeeery long")
    def test_valid_json(self):
        if _is_online():
            code = os.system(
                "ssh mkreik@duckstein -t "
                + "'bash -l -c /home/mkreik/bin/capture/status.sh' "
                + "| python -m json.tool > /dev/null")
            self.assertEqual(0, code)
        else:
            logging.warn("skipped test: not online")


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
    X = [(1, 0)] * size
    X.extend([(0, 1)] * size)
    if random:
        X = np.array(X).astype('float64')
        X += np.random.random_sample(X.shape) * 0.8 - 0.4
    y = [1] * size
    y.extend([-1] * size)
    return (X, y)


def _is_online():
    socket.setdefaulttimeout(1)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("8.8.8.8", 53))
        return True
    except socket.timeout:
        return False


if __name__ == '__main__':
    test_runner = TextTestRunner(resultclass=TimeLoggingTestResult)
    unittest.main(testRunner=test_runner)
