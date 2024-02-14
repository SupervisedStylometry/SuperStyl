import unittest
import superstyl.preproc.tuyau
import superstyl.preproc.features_extract

class DataLoading(unittest.TestCase):
    # First, testing the tuyau features
    def test_normalise(self):
        text = " Hello,  Mr. 𓀁, how are §§ you; doing?"
        expected_default = "hello mr how are you doing"
        self.assertEqual(superstyl.preproc.tuyau.normalise(text), expected_default)
        expected_keeppunct = "Hello, Mr. , how are SSSS you; doing?"
        self.assertEqual(superstyl.preproc.tuyau.normalise(text, keep_punct=True), expected_keeppunct)
        expected_keepsym = "Hello, Mr. 𓀁, how are §§ you; doing?"
        self.assertEqual(superstyl.preproc.tuyau.normalise(text, keep_sym=True), expected_keepsym)

    def test_detect_lang(self):
        french = "Bonjour, Monsieur, comment allez-vous?"
        english = "Hello, How do you do good sir?"
        italian = "Buongiorno signore, come sta?"
        #TODO: find something that manages old languages, like fasttext did…
        self.assertEqual(superstyl.preproc.tuyau.detect_lang(french), "fr")
        self.assertEqual(superstyl.preproc.tuyau.detect_lang(english), "en")
        self.assertEqual(superstyl.preproc.tuyau.detect_lang(italian), "it")

    # Now, lower level features,
    # from features_extract
    def test_counts(self):
        text = "the the the the"
        superstyl.preproc.features_extract.count_words(text, feat_list=None, feats = "words", n = 1, relFreqs = False)
        self.assertEqual(
            superstyl.preproc.features_extract.count_words(text, feat_list=None, feats = "words", n = 1, relFreqs = False),
            {'the': 4}
        )
        self.assertEqual(
            superstyl.preproc.features_extract.count_words(text, feat_list=None, feats="words", n=1, relFreqs=True),
            {'the': 1.0}
        )

        #TODO: a lot more tests



if __name__ == '__main__':
    unittest.main()
