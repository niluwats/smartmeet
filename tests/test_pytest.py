import tense
import unittest


class Test_TestCombineTense(unittest.TestCase):

    def test_combine_tense(self):
        self.assertEqual(tense.combine_sentence("hello"), 2)


if __name__ == '__main__':
    unittest.main()
