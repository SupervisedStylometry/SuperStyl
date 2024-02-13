import unittest
from superstyl.preproc.tuyau import normalise

class DataLoading(unittest.TestCase):
    def test_normalise(self):
        text = " Hello,  Mr. 𓀁, how are §§ you; doing?"
        expected_default = "hello mr how are you doing"
        self.assertEqual(normalise(text), expected_default)
        expected_keeppunct = "Hello, Mr. , how are SSSS you; doing?"
        self.assertEqual(normalise(text, keep_punct=True), expected_keeppunct)
        expected_keepsym = "Hello, Mr. 𓀁, how are §§ you; doing?"
        self.assertEqual(normalise(text, keep_sym=True), expected_keepsym)


if __name__ == '__main__':
    unittest.main()
