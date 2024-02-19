import unittest
import superstyl.preproc.tuyau
import superstyl.preproc.features_extract
import os
import glob

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class Main(unittest.TestCase):
    # FEATURE: from a list of paths, and several options, get a myTexts object, i.e., a list of dictionaries
    # for each text or samples, with metadata and the text itself
    # GIVEN
    paths = sorted(glob.glob(THIS_DIR + "/testdata/*.txt"))
    def test_load_texts_txt(self):
        # WHEN
        results = superstyl.preproc.tuyau.load_texts(self.paths, identify_lang=False, format="txt", keep_punct=False,
                                                    keep_sym=False, max_samples=None)
        # THEN
        expected = [{'name': 'Dupont_Letter1.txt', 'aut': 'Dupont', 'text': 'voici le texte', 'lang': 'NA'},
                    {'name': 'Smith_Letter1.txt', 'aut': 'Smith', 'text': 'this is the text', 'lang': 'NA'},
                    {'name': 'Smith_Letter2.txt', 'aut': 'Smith', 'text': 'this is also the text', 'lang': 'NA'}
                    ]

        self.assertEqual(results, expected)

        # WHEN
        results = superstyl.preproc.tuyau.load_texts(self.paths, identify_lang=False, format="txt", keep_punct=False,
                                                    keep_sym=False, max_samples=1)
        # THEN
        self.assertEqual(len([text for text in results if text["aut"] == 'Smith']), 1)

        # WHEN
        results = superstyl.preproc.tuyau.load_texts(self.paths, identify_lang=False, format="txt", keep_punct=True,
                                                     keep_sym=False, max_samples=None)
        # THEN
        expected = [{'name': 'Dupont_Letter1.txt', 'aut': 'Dupont', 'text': 'Voici le texte!', 'lang': 'NA'},
                    {'name': 'Smith_Letter1.txt', 'aut': 'Smith', 'text': 'This is the text!', 'lang': 'NA'},
                    {'name': 'Smith_Letter2.txt', 'aut': 'Smith', 'text': 'This is, also , the text!', 'lang': 'NA'}]

        self.assertEqual(results, expected)

        #TODO: test keep_sym, according to revised definition
        # WHEN
        # results = superstyl.preproc.tuyau.load_texts(self.paths, identify_lang=False, format="txt",
        #                                             keep_sym=True, max_samples=None)
        # THEN
        # expected = [{'name': 'Dupont_Letter1.txt', 'aut': 'Dupont', 'text': 'Voici le texte!', 'lang': 'NA'},
        #            {'name': 'Smith_Letter1.txt', 'aut': 'Smith', 'text': 'This is the text!', 'lang': 'NA'},
        #            {'name': 'Smith_Letter2.txt', 'aut': 'Smith', 'text': 'This is, also , the text!', 'lang': 'NA'}]

        # WHEN
        results = superstyl.preproc.tuyau.load_texts(self.paths, identify_lang=True, format="txt", keep_punct=True,
                                                     keep_sym=False, max_samples=None)
        # THEN
        # Just testing that a lang is predicted, not if it is ok or not
        self.assertEqual(len([text for text in results if text["lang"] != 'NA']), 3)

    #TODO: test other loading formats, that are not txt (and decide on their implementation)

    def test_docs_to_samples(self):
        # WHEN
        results = superstyl.preproc.tuyau.docs_to_samples(self.paths, identify_lang=False, size=2, step=None, units="words",
                                                feature="tokens", format="txt", keep_punct=False, keep_sym=False,
                                                max_samples=None)
        # THEN
        expected = [{'name': 'Dupont_Letter1.txt_0-2', 'aut': 'Dupont', 'text': 'voici le', 'lang': 'NA'},
                    {'name': 'Smith_Letter1.txt_0-2', 'aut': 'Smith', 'text': 'this is', 'lang': 'NA'},
                    {'name': 'Smith_Letter1.txt_2-4', 'aut': 'Smith', 'text': 'the text', 'lang': 'NA'},
                    {'name': 'Smith_Letter2.txt_0-2', 'aut': 'Smith', 'text': 'this is', 'lang': 'NA'},
                    {'name': 'Smith_Letter2.txt_2-4', 'aut': 'Smith', 'text': 'also the', 'lang': 'NA'}]
        self.assertEqual(results, expected)

        # WHEN
        results = superstyl.preproc.tuyau.docs_to_samples(self.paths, identify_lang=False, size=2, step=1,
                                                          units="words",
                                                          feature="tokens", format="txt", keep_punct=True,
                                                          keep_sym=False,
                                                          max_samples=None)

        # THEN
        expected = [{'name': 'Dupont_Letter1.txt_0-2', 'aut': 'Dupont', 'text': 'Voici le', 'lang': 'NA'},
                    {'name': 'Dupont_Letter1.txt_1-3', 'aut': 'Dupont', 'text': 'le texte', 'lang': 'NA'},
                    {'name': 'Dupont_Letter1.txt_2-4', 'aut': 'Dupont', 'text': 'texte !', 'lang': 'NA'},
                    {'name': 'Smith_Letter1.txt_0-2', 'aut': 'Smith', 'text': 'This is', 'lang': 'NA'},
                    {'name': 'Smith_Letter1.txt_1-3', 'aut': 'Smith', 'text': 'is the', 'lang': 'NA'},
                    {'name': 'Smith_Letter1.txt_2-4', 'aut': 'Smith', 'text': 'the text', 'lang': 'NA'},
                    {'name': 'Smith_Letter1.txt_3-5', 'aut': 'Smith', 'text': 'text !', 'lang': 'NA'},
                    {'name': 'Smith_Letter2.txt_0-2', 'aut': 'Smith', 'text': 'This is', 'lang': 'NA'},
                    {'name': 'Smith_Letter2.txt_1-3', 'aut': 'Smith', 'text': 'is ,', 'lang': 'NA'},
                    {'name': 'Smith_Letter2.txt_2-4', 'aut': 'Smith', 'text': ', also', 'lang': 'NA'},
                    {'name': 'Smith_Letter2.txt_3-5', 'aut': 'Smith', 'text': 'also ,', 'lang': 'NA'},
                    {'name': 'Smith_Letter2.txt_4-6', 'aut': 'Smith', 'text': ', the', 'lang': 'NA'},
                    {'name': 'Smith_Letter2.txt_5-7', 'aut': 'Smith', 'text': 'the text', 'lang': 'NA'},
                    {'name': 'Smith_Letter2.txt_6-8', 'aut': 'Smith', 'text': 'text !', 'lang': 'NA'}]
        self.assertEqual(results, expected)

        # TODO: test keep_sym

        # WHEN
        results = superstyl.preproc.tuyau.docs_to_samples(self.paths, identify_lang=True, size=2, step=None,
                                                          units="words",
                                                          feature="tokens", format="txt", keep_punct=False,
                                                          keep_sym=False,
                                                          max_samples=None)
        # THEN
        self.assertEqual(len([text for text in results if text["lang"] != 'NA']), 5)

        # WHEN
        results = superstyl.preproc.tuyau.docs_to_samples(self.paths, identify_lang=False, size=2, step=None,
                                                         units="words",
                                                         feature="tokens", format="txt", keep_punct=False,
                                                         keep_sym=False,
                                                         max_samples=1)
        # THEN
        self.assertEqual(len([text for text in results if text["aut"] == 'Smith']), 1)

    # TODO: test other loading formats with sampling, that are not txt (and decide on their implementation)

class DataLoading(unittest.TestCase):

     # Now down to lower level features
    # First, testing the tuyau features
    def test_normalise(self):
        text = " Hello,  Mr. 𓀁, how are §§ you; doing?"
        expected_default = "hello mr how are you doing"
        self.assertEqual(superstyl.preproc.tuyau.normalise(text), expected_default)
        expected_keeppunct = "Hello, Mr. , how are SSSS you; doing?"
        self.assertEqual(superstyl.preproc.tuyau.normalise(text, keep_punct=True), expected_keeppunct)
        expected_keepsym = "Hello, Mr. 𓀁, how are §§ you; doing?" #TODO: modify test according to new def
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

    def test_max_sampling(self):
        # FEATURE: randomly select a maximum number of samples by author/class
        # GIVEN
        myTexts = [
            {"name": "Letter1", "aut": "Smith", "text": "This is the text", "lang": "en"},
            {"name": "Letter2", "aut": "Smith", "text": "This is also the text", "lang": "en"},
            {"name": "Letter1", "aut": "Dupont", "text": "Voici le texte", "lang": "fr"},
        ]
        # WHEN
        results = superstyl.preproc.tuyau.max_sampling(myTexts, max_samples=1)
        # EXPECT
        self.assertEqual(len([text for text in results if text["aut"] == 'Smith']), 1)

    # Testing the processing of "myTexts" objects
    def test_get_feature_list(self):
        # FEATURE For a myTexts object with several texts, extract the relevant features (words or chars n-grams)
        # GIVEN
        myTexts = [
            {"name": "Letter1", "aut": "Smith", "text": "This is the text", "lang": "en"},
            {"name": "Letter2", "aut": "Smith", "text": "This is also the text", "lang": "en"},
            {"name": "Letter1", "aut": "Dupont", "text": "Voici le texte", "lang": "fr"},
        ]
        # WHEN
        results = superstyl.preproc.features_extract.get_feature_list(myTexts, feats="words", n=1, relFreqs=False)
        # THEN
        expected = [('This', 2), ('is', 2), ('the', 2), ('text', 2), ('also', 1), ('Voici', 1), ('le', 1), ('texte', 1)]
        self.assertEqual(results, expected)

        # WHEN
        results = superstyl.preproc.features_extract.get_feature_list(myTexts, feats="words", n=1, relFreqs=True)
        # THEN
        # TODO: BUG (sum of relative frequencies?)
        expected = [('This', 2/12), ('is', 2/12), ('the', 2/12), ('text', 2/12), ('Voici', 1/12),
                    ('le', 1/12), ('texte', 1/12), ('also', 1/12)]
        self.assertEqual(results, expected)

        # WHEN
        results = superstyl.preproc.features_extract.get_feature_list(myTexts, feats="words", n=2, relFreqs=False)

        # THEN
        expected = [('This_is', 2), ('the_text', 2), ('is_the', 1), ('is_also', 1), ('also_the', 1), ('Voici_le', 1), ('le_texte', 1)]
        self.assertEqual(results, expected)

        # WHEN
        results = superstyl.preproc.features_extract.get_feature_list(myTexts[0:2], feats="chars", n=2, relFreqs=False)

        # THEN
        expected = [('is', 4), ('s_', 4), ('_t', 4), ('Th', 2), ('hi', 2), ('_i', 2), ('th', 2), ('he', 2), ('e_', 2),
                    ('te', 2), ('ex', 2), ('xt', 2), ('_a', 1), ('al', 1), ('ls', 1), ('so', 1), ('o_', 1)]
        self.assertEqual(results, expected)




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


# TODO: tests for SVM, etc.
# Test all options of main commands, see if they are accepted or not

if __name__ == '__main__':
    unittest.main()
