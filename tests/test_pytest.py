import app
import unittest


class Test_TestDetection(unittest.TestCase):
    def test_mediapipe_detection(self):
        self.assertEqual(app.increment(3), 4)

    def test_decrement(self):
        self.assertEqual(inc_dec.decrement(3), 4)


if __name__ == '__main__':
    unittest.main()
