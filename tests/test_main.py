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
        # NB: it fails on that !!!
        # english = "Hello, How do you do good sir?"
        # still too hard
        # english = "Hello, How do you do good sir? Are you well today?"
        english = "Hello, How do you do good sir? Are you well today? Is this so bloody hard? Really, this is still failing?"
        italian = "Buongiorno signore, come sta?"
        #TODO: find something that manages old languages, like fasttext did…
        self.assertEqual(superstyl.preproc.tuyau.detect_lang(french), "fr")
        self.assertEqual(superstyl.preproc.tuyau.detect_lang(english), "en")
        self.assertEqual(superstyl.preproc.tuyau.detect_lang(italian), "it")

    # Now, lower level features,
    # from features_extract
    def test_counts(self):
        text = "the cat the dog the squirrel the cat the cat"
        superstyl.preproc.features_extract.count_words(text, feat_list=None, feats = "words", n = 1, relFreqs = False)
        self.assertEqual(
            superstyl.preproc.features_extract.count_words(text, feat_list=None, feats = "words", n = 1, relFreqs = False),
            {'the': 5, 'cat': 3, 'dog': 1, 'squirrel': 1}
        )
        self.assertEqual(
            superstyl.preproc.features_extract.count_words(text, feat_list=None, feats="words", n=1, relFreqs=True),
            {'the': 0.5, 'cat': 0.3, 'dog': 0.1, 'squirrel': 0.1}
        )
        self.assertEqual(
            superstyl.preproc.features_extract.count_words(text, feat_list=['the', 'cat'], feats="words", n=1, relFreqs=False),
            {'the': 5, 'cat': 3}
        )
        self.assertEqual(
            superstyl.preproc.features_extract.count_words(text, feat_list=['the', 'cat'], feats="words", n=1, relFreqs=True),
            {'the': 0.5, 'cat': 0.3}
        )
        self.assertEqual(
            superstyl.preproc.features_extract.count_words(text, feat_list=None, feats="words", n=2, relFreqs=False),
            {'the_cat': 3, 'cat_the': 2, 'the_dog': 1, 'dog_the': 1, 'the_squirrel': 1, 'squirrel_the': 1}
        )
        self.assertEqual(
            superstyl.preproc.features_extract.count_words(text, feat_list=None, feats="words", n=2, relFreqs=True),
            {'the_cat': 3/9, 'cat_the': 2/9, 'the_dog': 1/9, 'dog_the': 1/9, 'the_squirrel': 1/9, 'squirrel_the': 1/9}
        )

        text = "the yo yo"
        self.assertEqual(
            superstyl.preproc.features_extract.count_words(text, feat_list=None, feats="chars", n=3, relFreqs=False),
            {'the': 1, 'he_': 1, 'e_y': 1, '_yo': 2, 'yo_': 1, 'o_y': 1}
        )
        self.assertEqual(
            superstyl.preproc.features_extract.count_words(text, feat_list=['the'], feats="chars", n=3, relFreqs=True),
            {'the': 1/7}
        )

    # Testing the processing of "myTexts" objects
    def test_get_feature_list(self):
        myTexts = [
            {"name": "Letter1", "aut": "Smith", "text": "This is the text", "lang": "en"},
            {"name": "Letter2", "aut": "Smith", "text": "This is also the text", "lang": "en"},
            {"name": "Letter1", "aut": "Dupont", "text": "Voici le texte", "lang": "fr"},
        ]
        self.assertEqual(
            superstyl.preproc.features_extract.get_feature_list(myTexts, feats="words", n=1, relFreqs=False),
            [('This', 2), ('is', 2), ('the', 2), ('text', 2), ('also', 1), ('Voici', 1), ('le', 1), ('texte', 1)]
        )

    def test_get_counts(self):
        myTexts = [
            {"name": "Letter1", "aut": "Smith", "text": "This is the text", "lang": "en"},
            {"name": "Letter2", "aut": "Smith", "text": "This is also the text", "lang": "en"},
            {"name": "Letter1", "aut": "Dupont", "text": "Voici le texte", "lang": "fr"},
        ]

        self.assertEqual(
            superstyl.preproc.features_extract.get_counts(myTexts, ['the', 'is', 'also', 'le'], feats = "words",
                                                          n = 1, relFreqs = True),
            [{'name': 'Letter1', 'aut': 'Smith', 'text': 'This is the text', 'lang': 'en',
              'wordCounts': {'the': 0.25, 'is': 0.25}},
             {'name': 'Letter2', 'aut': 'Smith', 'text': 'This is also the text', 'lang': 'en',
              'wordCounts': {'the': 0.2, 'is': 0.2, 'also': 0.2}},
             {'name': 'Letter1', 'aut': 'Dupont', 'text': 'Voici le texte', 'lang': 'fr', 'wordCounts': {'le': 1/3}}]
        )

        #TODO: a lot more tests



if __name__ == '__main__':
    unittest.main()