import unittest

import counter

class TestCounter(unittest.TestCase):

    def setUp(self):
        pass

    def test__test(self):
        c = counter._test(35)
        self.assertTrue(c.timing)
        self.assertTrue(c.packets)

    def test_p_or_tiny(self):
        c_list = map(counter._test, [0, 2, 2, 2, 2, 2, 2, 4]) # length 8
        a = c_list[0]
        self.assertEqual(len(counter.p_or_tiny(c_list)), 7)
        self.assertEqual(len(c_list), 8, 'has side effect')
        self.assertEqual(c_list[0], a, 'has side effect')

    def test_p_or_toolong(self):
        c_list = map(counter._test, [2, 2, 2, 2, 2, 2, 4]) # length 7;no tiny
        self.assertEqual(len(counter.p_or_toolong(c_list)), 7)
        self.assertEqual(len(c_list), 7, 'has side effect')
        c_list2 = [counter._test(3, millisecs=4*60*1000)] # too long
        self.assertEqual(len(counter.p_or_toolong(c_list2)), 0)
        self.assertEqual(len(c_list2), 1, 'has side effect')
        c_list3 = [counter._test(3, val=10*60*1000)] # very big, but ok
        self.assertEqual(len(counter.p_or_toolong(c_list3)), 1)
        self.assertEqual(len(c_list3), 1, 'has side effect')

if __name__ == '__main__':
    unittest.main()
