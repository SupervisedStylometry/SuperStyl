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
        expected = [('This', 2/12), ('is', 2/12), ('the', 2/12), ('text', 2/12),  ('also', 1/12), ('Voici', 1/12),
                    ('le', 1/12), ('texte', 1/12)]
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

        # WHEN
        results = superstyl.preproc.features_extract.get_feature_list(myTexts[0:2], feats="chars", n=2, relFreqs=True)

        # THEN
        expected = [('is', 4/35), ('s_', 4/35), ('_t', 4/35), ('Th', 2/35), ('hi', 2/35), ('_i', 2/35), ('th', 2/35),
                    ('he', 2/35), ('e_', 2/35), ('te', 2/35), ('ex', 2/35), ('xt', 2/35), ('_a', 1/35), ('al', 1/35),
                    ('ls', 1/35), ('so', 1/35), ('o_', 1/35)]
        self.assertEqual(results, expected)


    def test_get_counts(self):
        # SCENARIO: given a myTexts object, i.e. a list of dictionaryies, containing metadata and text, count
        # the frequencies of features inside
        myTexts = [
            {"name": "Letter1", "aut": "Smith", "text": "This is the text", "lang": "en"},
            {"name": "Letter2", "aut": "Smith", "text": "This is also the text", "lang": "en"},
            {"name": "Letter1", "aut": "Dupont", "text": "Voici le texte", "lang": "fr"},
        ]
        # WHEN
        results = superstyl.preproc.features_extract.get_counts(myTexts, ['the', 'is', 'also', 'le'], feats = "words",
                                                          n = 1, relFreqs = True)
        # THEN
        expected = [{'name': 'Letter1', 'aut': 'Smith', 'text': 'This is the text', 'lang': 'en',
              'wordCounts': {'the': 0.25, 'is': 0.25}},
             {'name': 'Letter2', 'aut': 'Smith', 'text': 'This is also the text', 'lang': 'en',
              'wordCounts': {'the': 0.2, 'is': 0.2, 'also': 0.2}},
             {'name': 'Letter1', 'aut': 'Dupont', 'text': 'Voici le texte', 'lang': 'fr', 'wordCounts': {'le': 1/3}}]
        self.assertEqual(results, expected)

        # WHEN
        results = superstyl.preproc.features_extract.get_counts(myTexts, feats="words", n=1, relFreqs=True)
        # THEN
        expected = [{'name': 'Letter1', 'aut': 'Smith', 'text': 'This is the text', 'lang': 'en',
                     'wordCounts': {'This': 0.25, 'is': 0.25, 'the': 0.25, 'text': 0.25}},
                   {'name': 'Letter2', 'aut': 'Smith', 'text': 'This is also the text', 'lang': 'en', 'wordCounts':
                       {'This': 0.2, 'is': 0.2, 'also': 0.2, 'the': 0.2, 'text': 0.2}},
                    {'name': 'Letter1', 'aut': 'Dupont', 'text': 'Voici le texte', 'lang': 'fr', 'wordCounts':
                        {'Voici': 1/3, 'le': 1/3, 'texte': 1/3}}]
        self.assertEqual(results, expected)

        # WHEN
        results = superstyl.preproc.features_extract.get_counts(myTexts, feats="words", n=2, relFreqs=False)
        # THEN
        expected = [{'name': 'Letter1', 'aut': 'Smith', 'text': 'This is the text', 'lang': 'en', 'wordCounts':
            {'This_is': 1, 'is_the': 1, 'the_text': 1}},
                    {'name': 'Letter2', 'aut': 'Smith', 'text': 'This is also the text', 'lang': 'en', 'wordCounts':
                        {'This_is': 1, 'is_also': 1, 'also_the': 1, 'the_text': 1}},
                    {'name': 'Letter1', 'aut': 'Dupont', 'text': 'Voici le texte', 'lang': 'fr', 'wordCounts':
                        {'Voici_le': 1, 'le_texte': 1}}]
        self.assertEqual(results, expected)

        # WHEN
        results = superstyl.preproc.features_extract.get_counts(myTexts, feats="words", n=2, relFreqs=True)
        # THEN
        expected = [{'name': 'Letter1', 'aut': 'Smith', 'text': 'This is the text', 'lang': 'en', 'wordCounts':
            {'This_is': 1/3, 'is_the': 1/3, 'the_text': 1/3}},
                    {'name': 'Letter2', 'aut': 'Smith', 'text': 'This is also the text', 'lang': 'en', 'wordCounts':
                        {'This_is': 1/4, 'is_also': 1/4, 'also_the': 1/4, 'the_text': 1/4}},
                    {'name': 'Letter1', 'aut': 'Dupont', 'text': 'Voici le texte', 'lang': 'fr', 'wordCounts':
                        {'Voici_le': 1/2, 'le_texte': 1/2}}]
        self.assertEqual(results, expected)

        # WHEN
        results = superstyl.preproc.features_extract.get_counts(myTexts, feat_list=["This_is", "le_texte"],
                                                                feats="words", n=2, relFreqs=True)
        # THEN
        expected = [{'name': 'Letter1', 'aut': 'Smith', 'text': 'This is the text', 'lang': 'en', 'wordCounts':
            {'This_is': 1 / 3}},
                    {'name': 'Letter2', 'aut': 'Smith', 'text': 'This is also the text', 'lang': 'en', 'wordCounts':
                        {'This_is': 1/4}},
                    {'name': 'Letter1', 'aut': 'Dupont', 'text': 'Voici le texte', 'lang': 'fr', 'wordCounts':
                        {'le_texte': 1 / 2}}]
        self.assertEqual(results, expected)

        # WHEN
        results = superstyl.preproc.features_extract.get_counts(myTexts, feat_list=["th"],
                                                                feats="chars", n=2, relFreqs=True)
        # THEN
        expected = [{'name': 'Letter1', 'aut': 'Smith', 'text': 'This is the text', 'lang': 'en',
                     'wordCounts': {'th': 1/15}},
                    {'name': 'Letter2', 'aut': 'Smith', 'text': 'This is also the text', 'lang': 'en',
                     'wordCounts': {'th': 1/20}},
                    {'name': 'Letter1', 'aut': 'Dupont', 'text': 'Voici le texte', 'lang': 'fr', 'wordCounts': {}}]

        self.assertEqual(results, expected)

    # TODO: test get_embedded_counts and load_embedding

    # TODO: test count_process

    # TODO: test features_select
    # TODO: test select


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
        # Scenario: given a text, extract a list of the features that appear in it, with their counts in absolute frequency
        # GIVEN
        text = "the cat the dog the squirrel the cat the cat"
        # WHEN
        results = superstyl.preproc.features_extract.count_words(text, feats = "words", n = 1)
        # THEN
        expected = {'the': 5, 'cat': 3, 'dog': 1, 'squirrel': 1}
        self.assertEqual(results, expected)

        # WHEN
        results = superstyl.preproc.features_extract.count_words(text, feats="words", n=2)
        # THEN
        expected = {'the_cat': 3, 'cat_the': 2, 'the_dog': 1, 'dog_the': 1, 'the_squirrel': 1, 'squirrel_the': 1}
        self.assertEqual(results, expected)

        # GIVEN
        text = "the yo yo"
        # WHEN
        results = superstyl.preproc.features_extract.count_words(text, feats="chars", n=3)
        # THEN
        expected = {'the': 1, 'he_': 1, 'e_y': 1, '_yo': 2, 'yo_': 1, 'o_y': 1}
        self.assertEqual(results, expected)

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


# TODO: tests for SVM, etc.
# Test all options of main commands, see if they are accepted or not

if __name__ == '__main__':
    unittest.main()
