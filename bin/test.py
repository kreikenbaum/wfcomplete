import unittest

import counter

class TestCounter(unittest.TestCase):

    def setUp(self):
    
    def test__test(self):
        c = counter._test(35)
        self.assertTrue(c.timing)
        self.assertTrue(c.packets)

    def test_p_or_tiny(self):
        c_list = map(counter._test, [1, 2, 2, 2, 2, 3, 4]) # length 7
        with_0 = c_list[:]
        with_0.append(counter._test(0))
        a = c_list[0]
        self.assertEqual(len(counter.p_or_tiny(c_list)), 6)
        self.assertEqual(len(counter.p_or_tiny(with_0)), 6)
        self.assertEqual(len(with_0), 8, 'has side effect')
        self.assertEqual(len(c_list), 7, 'has side effect')
        self.assertEqual(c_list[0], a, 'has side effect')

    def test_p_or_toolong(self):
        too_long = [counter._test(3, millisecs=4*60*1000)]
        big_val = [counter._test(3, val=10*60*1000)] # very big, but ok
        self.assertEqual(len(counter.p_or_toolong(c_list)), 7)
        self.assertEqual(len(c_list), 7, 'has side effect')
        self.assertEqual(len(counter.p_or_toolong(too_long)), 0)
        self.assertEqual(len(too_long), 1, 'has side effect')
        self.assertEqual(len(counter.p_or_toolong(big_val)), 1)
        self.assertEqual(len(big_val), 1, 'has side effect')

    def test_outlier_removal(self):
        # take simple dict which removes stuff
        # by methods 1-3 (one each)
        # check that original is still the same
        c_dict = {'url': self.c_list }
        c_dict['url'].extend(self.big_val)
        self.assertEqual(len(counter.outlier_removal(c_dict, 1)['url']), 7)
        self.assertEqual(len(c_dict['url']), 8, 'has side effect')
        self.assertEqual(len(counter.outlier_removal(c_dict, 1)['url']), 7)
        
        

if __name__ == '__main__':
    unittest.main()
