#! /usr/bin/env python
import unittest

import analyse
import counter

class TestCounter(unittest.TestCase):

    def setUp(self):
        self.c_list = map(counter._test, [1, 2, 2, 2, 2, 3, 4]) # length 7
        self.big_val = [counter._test(3, val=10*60*1000)] # very big, but ok

    def test__test(self):
        c = counter._test(35)
        self.assertTrue(c.timing)
        self.assertTrue(c.packets)

    def test_p_or_tiny(self):
        with_0 = self.c_list[:]
        with_0.append(counter._test(0))
        a = self.c_list[0]
        self.assertEqual(len(counter.p_or_tiny(self.c_list)), 6)
        self.assertEqual(len(counter.p_or_tiny(with_0)), 6)
        self.assertEqual(len(with_0), 8, 'has side effect')
        self.assertEqual(len(self.c_list), 7, 'has side effect')
        self.assertEqual(self.c_list[0], a, 'has side effect')

    def test_p_or_toolong(self):
        too_long = [counter._test(3, millisecs=4*60*1000)]
        self.assertEqual(len(counter.p_or_toolong(self.c_list)), 7)
        self.assertEqual(len(self.c_list), 7, 'has side effect')
        self.assertEqual(len(counter.p_or_toolong(too_long)), 0)
        self.assertEqual(len(too_long), 1, 'has side effect')
        self.assertEqual(len(counter.p_or_toolong(self.big_val)), 1)
        self.assertEqual(len(self.big_val), 1, 'has side effect')

    def test_outlier_removal(self):
        # take simple dict which removes stuff
        # by methods 1-3 (one each)
        # check that original is still the same
        c_dict = {'url': self.c_list }
        c_dict['url'].extend(self.big_val)
        self.assertEqual(len(counter.outlier_removal(c_dict, 1)['url']), 7)
        self.assertEqual(len(c_dict['url']), 8, 'has side effect')
        self.assertEqual(len(counter.outlier_removal(c_dict, 1)['url']), 7)
        
class TestAnalyse(unittest.TestCase):

    def setUp(self):
        self.base_mock = {'a': (10, -1), 'b': (10, -1)}
        self.base_mock2 = {'a': (10, -1), 'b': (10, -1), 'c': (10, -1)}

    def test__size_increase_equal(self):
        self.assertEqual(analyse._size_increase(self.base_mock,
                                                {'a': (10, -1), 'b': (10, -1)}),
                         100)
    def test__size_increase_same_half(self):
        self.assertEqual(analyse._size_increase(self.base_mock,
                                                {'a': (5, -1), 'b': (5, -1)}),
                         50)

    def test__size_increase_same_double(self):
        self.assertEqual(analyse._size_increase(self.base_mock,
                                                {'a': (20, -1), 'b': (20, -1)}),
                         200)

    def test__size_increase_one_double(self):
        self.assertAlmostEqual(analyse._size_increase(self.base_mock,
                                                {'a': (10, -1), 'b': (20, -1)}),
                         100*pow(2, 1./2))

    def test__size_increase_both_different(self):
        self.assertEqual(analyse._size_increase(self.base_mock,
                                                {'a': (5, -1), 'b': (20, -1)}),
                         100)

    def test__size_increase_three_one(self):
        self.assertAlmostEqual(analyse._size_increase(
            self.base_mock2, {'a': (10, -1), 'b': (10, -1), 'c': (20, -1)}),
                               100.*pow(2, 1./3))

    def test__size_increase_three_one_reverted(self):
        self.assertAlmostEqual(analyse._size_increase(
            {'a': (10, -1), 'b': (10, -1), 'c': (20, -1)}, self.base_mock2),
                               100.*pow(1./2, 1./3))

if __name__ == '__main__':
    unittest.main()
