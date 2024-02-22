import unittest
import superstyl
import superstyl.preproc
import superstyl.preproc.pipe
import superstyl.preproc.features_extract
import superstyl.preproc.embedding
import superstyl.preproc.select
import superstyl.preproc.text_count
import os
import glob

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class Main(unittest.TestCase):
    # FEATURE: from a list of paths, and several options, get a corpus, composed of a pandas table of metadata and counts,
    # as well as a list of feats
    # GIVEN
    paths = sorted(glob.glob(THIS_DIR + "/testdata/*.txt"))

    def test_load_corpus(self):
        # WHEN
        corpus, feats = superstyl.load.load_corpus(self.paths)
        # THEN
        expected_feats = [('this', 2/12), ('is', 2/12), ('the', 2/12), ('text', 2/12), ('voici', 1/12),
                    ('le', 1/12), ('texte', 1/12), ('also', 1/12)]
        expected_corpus = {'author': {'Dupont_Letter1.txt': 'Dupont', 'Smith_Letter1.txt': 'Smith', 'Smith_Letter2.txt': 'Smith'},
                           'lang': {'Dupont_Letter1.txt': 'NA', 'Smith_Letter1.txt': 'NA', 'Smith_Letter2.txt': 'NA'},
                           'this': {'Dupont_Letter1.txt': 0.0, 'Smith_Letter1.txt': 0.25, 'Smith_Letter2.txt': 0.2},
                           'is': {'Dupont_Letter1.txt': 0.0, 'Smith_Letter1.txt': 0.25, 'Smith_Letter2.txt': 0.2},
                           'the': {'Dupont_Letter1.txt': 0.0, 'Smith_Letter1.txt': 0.25, 'Smith_Letter2.txt': 0.2},
                           'text': {'Dupont_Letter1.txt': 0.0, 'Smith_Letter1.txt': 0.25, 'Smith_Letter2.txt': 0.2},
                           'voici': {'Dupont_Letter1.txt': 1/3, 'Smith_Letter1.txt': 0.0, 'Smith_Letter2.txt': 0.0},
                           'le': {'Dupont_Letter1.txt': 1/3, 'Smith_Letter1.txt': 0.0, 'Smith_Letter2.txt': 0.0},
                           'texte': {'Dupont_Letter1.txt': 1/3, 'Smith_Letter1.txt': 0.0, 'Smith_Letter2.txt': 0.0},
                           'also': {'Dupont_Letter1.txt': 0.0, 'Smith_Letter1.txt': 0.0, 'Smith_Letter2.txt': 0.2}}
        self.assertEqual(feats, expected_feats)
        self.assertEqual(corpus.to_dict(), expected_corpus)

        # WHEN
        corpus, feats = superstyl.load.load_corpus(self.paths, feat_list=[('the', 0)], feats="chars", n=3, k=5000, relFreqs=False,
                                                   format="txt", keep_punct=False, keep_sym=False, identify_lang=True)

        # THEN
        expected_feats = [('the', 0)]
        # TODO: improve langage identification so we don't have to plan for error…
        expected_corpus = {'author': {'Dupont_Letter1.txt': 'Dupont', 'Smith_Letter1.txt': 'Smith', 'Smith_Letter2.txt': 'Smith'},
                           'lang': {'Dupont_Letter1.txt': 'ro', 'Smith_Letter1.txt': 'en', 'Smith_Letter2.txt': 'en'},
                           'the': {'Dupont_Letter1.txt': 0, 'Smith_Letter1.txt': 1, 'Smith_Letter2.txt': 1}}

        self.assertEqual(feats, expected_feats)
        self.assertEqual(corpus.to_dict(), expected_corpus)

        # WHEN
        corpus, feats = superstyl.load.load_corpus(self.paths, feats="words", n=1,
                                                   sampling=True, units="words", size=2, step=None,
                                                   keep_punct=True, keep_sym=False)

        # THEN
        expected_feats = [('!', 2/16), ('This', 2/16), ('is', 2/16), ('the', 2/16), ('text', 2/16), (',', 2/16),
                          ('Voici', 1/16), ('le', 1/16), ('texte', 1/16), ('also', 1/16)]

        expected_corpus = {'author': {'Dupont_Letter1.txt_0-2': 'Dupont', 'Dupont_Letter1.txt_2-4': 'Dupont',
                                      'Smith_Letter1.txt_0-2': 'Smith', 'Smith_Letter1.txt_2-4': 'Smith',
                                      'Smith_Letter2.txt_0-2': 'Smith', 'Smith_Letter2.txt_2-4': 'Smith',
                                      'Smith_Letter2.txt_4-6': 'Smith', 'Smith_Letter2.txt_6-8': 'Smith'},
                           'lang': {'Dupont_Letter1.txt_0-2': 'NA', 'Dupont_Letter1.txt_2-4': 'NA',
                                    'Smith_Letter1.txt_0-2': 'NA', 'Smith_Letter1.txt_2-4': 'NA',
                                    'Smith_Letter2.txt_0-2': 'NA', 'Smith_Letter2.txt_2-4': 'NA',
                                    'Smith_Letter2.txt_4-6': 'NA', 'Smith_Letter2.txt_6-8': 'NA'},
                           '!': {'Dupont_Letter1.txt_0-2': 0.0, 'Dupont_Letter1.txt_2-4': 0.5,
                                 'Smith_Letter1.txt_0-2': 0.0, 'Smith_Letter1.txt_2-4': 0.0,
                                 'Smith_Letter2.txt_0-2': 0.0, 'Smith_Letter2.txt_2-4': 0.0,
                                 'Smith_Letter2.txt_4-6': 0.0, 'Smith_Letter2.txt_6-8': 0.5},
                           'This': {'Dupont_Letter1.txt_0-2': 0.0, 'Dupont_Letter1.txt_2-4': 0.0,
                                    'Smith_Letter1.txt_0-2': 0.5, 'Smith_Letter1.txt_2-4': 0.0,
                                    'Smith_Letter2.txt_0-2': 0.5, 'Smith_Letter2.txt_2-4': 0.0,
                                    'Smith_Letter2.txt_4-6': 0.0, 'Smith_Letter2.txt_6-8': 0.0},
                           'is': {'Dupont_Letter1.txt_0-2': 0.0, 'Dupont_Letter1.txt_2-4': 0.0,
                                  'Smith_Letter1.txt_0-2': 0.5, 'Smith_Letter1.txt_2-4': 0.0,
                                  'Smith_Letter2.txt_0-2': 0.5, 'Smith_Letter2.txt_2-4': 0.0,
                                  'Smith_Letter2.txt_4-6': 0.0, 'Smith_Letter2.txt_6-8': 0.0},
                           'the': {'Dupont_Letter1.txt_0-2': 0.0, 'Dupont_Letter1.txt_2-4': 0.0,
                                   'Smith_Letter1.txt_0-2': 0.0, 'Smith_Letter1.txt_2-4': 0.5,
                                   'Smith_Letter2.txt_0-2': 0.0, 'Smith_Letter2.txt_2-4': 0.0,
                                   'Smith_Letter2.txt_4-6': 0.5, 'Smith_Letter2.txt_6-8': 0.0},
                           'text': {'Dupont_Letter1.txt_0-2': 0.0, 'Dupont_Letter1.txt_2-4': 0.0,
                                    'Smith_Letter1.txt_0-2': 0.0, 'Smith_Letter1.txt_2-4': 0.5,
                                    'Smith_Letter2.txt_0-2': 0.0, 'Smith_Letter2.txt_2-4': 0.0,
                                    'Smith_Letter2.txt_4-6': 0.0, 'Smith_Letter2.txt_6-8': 0.5},
                           ',': {'Dupont_Letter1.txt_0-2': 0.0, 'Dupont_Letter1.txt_2-4': 0.0,
                                 'Smith_Letter1.txt_0-2': 0.0, 'Smith_Letter1.txt_2-4': 0.0,
                                 'Smith_Letter2.txt_0-2': 0.0, 'Smith_Letter2.txt_2-4': 0.5,
                                 'Smith_Letter2.txt_4-6': 0.5, 'Smith_Letter2.txt_6-8': 0.0},
                           'Voici': {'Dupont_Letter1.txt_0-2': 0.5, 'Dupont_Letter1.txt_2-4': 0.0,
                                     'Smith_Letter1.txt_0-2': 0.0, 'Smith_Letter1.txt_2-4': 0.0,
                                     'Smith_Letter2.txt_0-2': 0.0, 'Smith_Letter2.txt_2-4': 0.0,
                                     'Smith_Letter2.txt_4-6': 0.0, 'Smith_Letter2.txt_6-8': 0.0},
                           'le': {'Dupont_Letter1.txt_0-2': 0.5, 'Dupont_Letter1.txt_2-4': 0.0,
                                  'Smith_Letter1.txt_0-2': 0.0, 'Smith_Letter1.txt_2-4': 0.0,
                                  'Smith_Letter2.txt_0-2': 0.0, 'Smith_Letter2.txt_2-4': 0.0,
                                  'Smith_Letter2.txt_4-6': 0.0, 'Smith_Letter2.txt_6-8': 0.0},
                           'texte': {'Dupont_Letter1.txt_0-2': 0.0, 'Dupont_Letter1.txt_2-4': 0.5,
                                     'Smith_Letter1.txt_0-2': 0.0, 'Smith_Letter1.txt_2-4': 0.0,
                                     'Smith_Letter2.txt_0-2': 0.0, 'Smith_Letter2.txt_2-4': 0.0,
                                     'Smith_Letter2.txt_4-6': 0.0, 'Smith_Letter2.txt_6-8': 0.0},
                           'also': {'Dupont_Letter1.txt_0-2': 0.0, 'Dupont_Letter1.txt_2-4': 0.0,
                                    'Smith_Letter1.txt_0-2': 0.0, 'Smith_Letter1.txt_2-4': 0.0,
                                    'Smith_Letter2.txt_0-2': 0.0, 'Smith_Letter2.txt_2-4': 0.5,
                                    'Smith_Letter2.txt_4-6': 0.0, 'Smith_Letter2.txt_6-8': 0.0}}


        self.assertEqual(feats, expected_feats)
        self.assertEqual(corpus.to_dict(), expected_corpus)

        # WHEN
        corpus, feats = superstyl.load.load_corpus(self.paths, k=4)
        # THEN
        expected_feats = [('this', 2 / 12), ('is', 2 / 12), ('the', 2 / 12), ('text', 2 / 12)]
        self.assertEqual(feats, expected_feats)

        # WHEN
        corpus, feats = superstyl.load.load_corpus(self.paths, feats="chars", n=3, format="txt", keep_punct=True,
                                                   relFreqs=False)

        # THEN
        expected_feats = [('e_t', 3), ('_te', 3), ('tex', 3), ('ext', 3), ('is_', 3), ('Thi', 2), ('his', 2), ('s_i', 2),
                          ('_is', 2), ('_th', 2), ('the', 2), ('he_', 2), ('xt!', 2), ('Voi', 1), ('oic', 1), ('ici', 1),
                          ('ci_', 1), ('i_l', 1), ('_le', 1), ('le_', 1), ('xte', 1), ('te!', 1), ('is,', 1), ('s,_', 1),
                          (',_a', 1), ('_al', 1), ('als', 1), ('lso', 1), ('so_', 1), ('o_,', 1), ('_,_', 1), (',_t', 1),
                          ('s_t', 1)]


        expected_corpus = {'author':
                               {'Dupont_Letter1.txt': 'Dupont', 'Smith_Letter2.txt': 'Smith', 'Smith_Letter1.txt': 'Smith'},
                           'lang': {'Dupont_Letter1.txt': 'NA', 'Smith_Letter2.txt': 'NA', 'Smith_Letter1.txt': 'NA'},
                           'e_t': {'Dupont_Letter1.txt': 1, 'Smith_Letter2.txt': 1, 'Smith_Letter1.txt': 1},
                           '_te': {'Dupont_Letter1.txt': 1, 'Smith_Letter2.txt': 1, 'Smith_Letter1.txt': 1},
                           'tex': {'Dupont_Letter1.txt': 1, 'Smith_Letter2.txt': 1, 'Smith_Letter1.txt': 1},
                           'ext': {'Dupont_Letter1.txt': 1, 'Smith_Letter2.txt': 1, 'Smith_Letter1.txt': 1},
                           'is_': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1, 'Smith_Letter1.txt': 2},
                           'Thi': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1, 'Smith_Letter1.txt': 1},
                           'his': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1, 'Smith_Letter1.txt': 1},
                           's_i': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1, 'Smith_Letter1.txt': 1},
                           '_is': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1, 'Smith_Letter1.txt': 1},
                           '_th': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1, 'Smith_Letter1.txt': 1},
                           'the': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1, 'Smith_Letter1.txt': 1},
                           'he_': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1, 'Smith_Letter1.txt': 1},
                           'xt!': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1, 'Smith_Letter1.txt': 1},
                           'Voi': {'Dupont_Letter1.txt': 1, 'Smith_Letter2.txt': 0, 'Smith_Letter1.txt': 0},
                           'oic': {'Dupont_Letter1.txt': 1, 'Smith_Letter2.txt': 0, 'Smith_Letter1.txt': 0},
                           'ici': {'Dupont_Letter1.txt': 1, 'Smith_Letter2.txt': 0, 'Smith_Letter1.txt': 0},
                           'ci_': {'Dupont_Letter1.txt': 1, 'Smith_Letter2.txt': 0, 'Smith_Letter1.txt': 0},
                           'i_l': {'Dupont_Letter1.txt': 1, 'Smith_Letter2.txt': 0, 'Smith_Letter1.txt': 0},
                           '_le': {'Dupont_Letter1.txt': 1, 'Smith_Letter2.txt': 0, 'Smith_Letter1.txt': 0},
                           'le_': {'Dupont_Letter1.txt': 1, 'Smith_Letter2.txt': 0, 'Smith_Letter1.txt': 0},
                           'xte': {'Dupont_Letter1.txt': 1, 'Smith_Letter2.txt': 0, 'Smith_Letter1.txt': 0},
                           'te!': {'Dupont_Letter1.txt': 1, 'Smith_Letter2.txt': 0, 'Smith_Letter1.txt': 0},
                           'is,': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1, 'Smith_Letter1.txt': 0},
                           's,_': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1, 'Smith_Letter1.txt': 0},
                           ',_a': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1, 'Smith_Letter1.txt': 0},
                           '_al': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1, 'Smith_Letter1.txt': 0},
                           'als': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1, 'Smith_Letter1.txt': 0},
                           'lso': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1, 'Smith_Letter1.txt': 0},
                           'so_': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1, 'Smith_Letter1.txt': 0},
                           'o_,': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1, 'Smith_Letter1.txt': 0},
                           '_,_': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1, 'Smith_Letter1.txt': 0},
                           ',_t': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1, 'Smith_Letter1.txt': 0},
                           's_t': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 0, 'Smith_Letter1.txt': 1}}


        self.assertEqual(sorted(feats), sorted(expected_feats))
        self.assertEqual(corpus.to_dict(), expected_corpus)

        # WHEN
        corpus, feats = superstyl.load.load_corpus(self.paths, feats="affixes", n=3, format="txt", keep_punct=True)

        # THEN
        expected_feats = [('_te', 3/51), ('tex', 3/51), ('ext', 2/51), ('is_', 3/51), ('Thi', 2/51), ('his', 2/51),
                          ('_is', 2/51), ('_th', 2/51),  ('he_', 2/51), ('xt!', 2/51), ('Voi', 1/51),('ici', 1/51),
                          ('ci_', 1/51), ('_le', 1/51), ('le_', 1/51), ('xte', 1/51), ('te!', 1/51), ('is,', 1/51),
                          ('s,_', 1/51), (',_a', 1/51), ('_al', 1/51), ('als', 1/51), ('lso', 1/51), ('so_', 1/51),
                          ('o_,', 1/51), ('_,_', 1/51), (',_t', 1/51)]

        expected_corpus = {'author':
                               {'Dupont_Letter1.txt': 'Dupont', 'Smith_Letter2.txt': 'Smith',
                                'Smith_Letter1.txt': 'Smith'},
                           'lang': {'Dupont_Letter1.txt': 'NA', 'Smith_Letter2.txt': 'NA', 'Smith_Letter1.txt': 'NA'},
                           '_te': {'Dupont_Letter1.txt': 1/13, 'Smith_Letter2.txt': 1/23, 'Smith_Letter1.txt': 1/15},
                           'tex': {'Dupont_Letter1.txt': 1/13, 'Smith_Letter2.txt': 1/23, 'Smith_Letter1.txt': 1/15},
                           'ext': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1/23, 'Smith_Letter1.txt': 1/15},
                           'is_': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1/23, 'Smith_Letter1.txt': 2/15},
                           'Thi': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1/23, 'Smith_Letter1.txt': 1/15},
                           'his': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1/23, 'Smith_Letter1.txt': 1/15},
                           '_is': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1/23, 'Smith_Letter1.txt': 1/15},
                           '_th': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1/23, 'Smith_Letter1.txt': 1/15},
                           'he_': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1/23, 'Smith_Letter1.txt': 1/15},
                           'xt!': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1/23, 'Smith_Letter1.txt': 1/15},
                           'Voi': {'Dupont_Letter1.txt': 1/13, 'Smith_Letter2.txt': 0, 'Smith_Letter1.txt': 0},
                           'ici': {'Dupont_Letter1.txt': 1/13, 'Smith_Letter2.txt': 0, 'Smith_Letter1.txt': 0},
                           'ci_': {'Dupont_Letter1.txt': 1/13, 'Smith_Letter2.txt': 0, 'Smith_Letter1.txt': 0},
                           '_le': {'Dupont_Letter1.txt': 1/13, 'Smith_Letter2.txt': 0, 'Smith_Letter1.txt': 0},
                           'le_': {'Dupont_Letter1.txt': 1/13, 'Smith_Letter2.txt': 0, 'Smith_Letter1.txt': 0},
                           'xte': {'Dupont_Letter1.txt': 1/13, 'Smith_Letter2.txt': 0, 'Smith_Letter1.txt': 0},
                           'te!': {'Dupont_Letter1.txt': 1/13, 'Smith_Letter2.txt': 0, 'Smith_Letter1.txt': 0},
                           'is,': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1/23, 'Smith_Letter1.txt': 0},
                           's,_': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1/23, 'Smith_Letter1.txt': 0},
                           ',_a': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1/23, 'Smith_Letter1.txt': 0},
                           '_al': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1/23, 'Smith_Letter1.txt': 0},
                           'als': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1/23, 'Smith_Letter1.txt': 0},
                           'lso': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1/23, 'Smith_Letter1.txt': 0},
                           'so_': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1/23, 'Smith_Letter1.txt': 0},
                           'o_,': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1/23, 'Smith_Letter1.txt': 0},
                           '_,_': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1/23, 'Smith_Letter1.txt': 0},
                           ',_t': {'Dupont_Letter1.txt': 0, 'Smith_Letter2.txt': 1/23, 'Smith_Letter1.txt': 0}}

        self.assertEqual(sorted(feats), sorted(expected_feats))
        self.assertEqual(corpus.to_dict(), expected_corpus)


    def test_load_texts_txt(self):
        # SCENARIO: from paths to txt, get myTexts object, i.e., a list of dictionaries
        #     # for each text or samples, with metadata and the text itself
        # WHEN
        results = superstyl.preproc.pipe.load_texts(self.paths, identify_lang=False, format="txt", keep_punct=False,
                                                    keep_sym=False, max_samples=None)
        # THEN
        expected = [{'name': 'Dupont_Letter1.txt', 'aut': 'Dupont', 'text': 'voici le texte', 'lang': 'NA'},
                    {'name': 'Smith_Letter1.txt', 'aut': 'Smith', 'text': 'this is the text', 'lang': 'NA'},
                    {'name': 'Smith_Letter2.txt', 'aut': 'Smith', 'text': 'this is also the text', 'lang': 'NA'}
                    ]

        self.assertEqual(results, expected)

        # WHEN
        results = superstyl.preproc.pipe.load_texts(self.paths, identify_lang=False, format="txt", keep_punct=False,
                                                    keep_sym=False, max_samples=1)
        # THEN
        self.assertEqual(len([text for text in results if text["aut"] == 'Smith']), 1)

        # WHEN
        results = superstyl.preproc.pipe.load_texts(self.paths, identify_lang=False, format="txt", keep_punct=True,
                                                     keep_sym=False, max_samples=None)
        # THEN
        expected = [{'name': 'Dupont_Letter1.txt', 'aut': 'Dupont', 'text': 'Voici le texte!', 'lang': 'NA'},
                    {'name': 'Smith_Letter1.txt', 'aut': 'Smith', 'text': 'This is the text!', 'lang': 'NA'},
                    {'name': 'Smith_Letter2.txt', 'aut': 'Smith', 'text': 'This is, also , the text!', 'lang': 'NA'}]

        self.assertEqual(results, expected)

        #TODO: test keep_sym, according to revised definition
        # WHEN
        # results = superstyl.preproc.pipe.load_texts(self.paths, identify_lang=False, format="txt",
        #                                             keep_sym=True, max_samples=None)
        # THEN
        # expected = [{'name': 'Dupont_Letter1.txt', 'aut': 'Dupont', 'text': 'Voici le texte!', 'lang': 'NA'},
        #            {'name': 'Smith_Letter1.txt', 'aut': 'Smith', 'text': 'This is the text!', 'lang': 'NA'},
        #            {'name': 'Smith_Letter2.txt', 'aut': 'Smith', 'text': 'This is, also , the text!', 'lang': 'NA'}]

        # WHEN
        results = superstyl.preproc.pipe.load_texts(self.paths, identify_lang=True, format="txt", keep_punct=True,
                                                     keep_sym=False, max_samples=None)
        # THEN
        # Just testing that a lang is predicted, not if it is ok or not
        self.assertEqual(len([text for text in results if text["lang"] != 'NA']), 3)

    #TODO: test other loading formats, that are not txt (and decide on their implementation)

    def test_docs_to_samples(self):
        # WHEN
        results = superstyl.preproc.pipe.docs_to_samples(self.paths, identify_lang=False, size=2, step=None, units="words",
                                                format="txt", keep_punct=False, keep_sym=False, max_samples=None)
        # THEN
        expected = [{'name': 'Dupont_Letter1.txt_0-2', 'aut': 'Dupont', 'text': 'voici le', 'lang': 'NA'},
                    {'name': 'Smith_Letter1.txt_0-2', 'aut': 'Smith', 'text': 'this is', 'lang': 'NA'},
                    {'name': 'Smith_Letter1.txt_2-4', 'aut': 'Smith', 'text': 'the text', 'lang': 'NA'},
                    {'name': 'Smith_Letter2.txt_0-2', 'aut': 'Smith', 'text': 'this is', 'lang': 'NA'},
                    {'name': 'Smith_Letter2.txt_2-4', 'aut': 'Smith', 'text': 'also the', 'lang': 'NA'}]
        self.assertEqual(results, expected)

        # WHEN
        results = superstyl.preproc.pipe.docs_to_samples(self.paths, identify_lang=False, size=2, step=1,
                                                          units="words", format="txt", keep_punct=True,
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
        results = superstyl.preproc.pipe.docs_to_samples(self.paths, identify_lang=True, size=2, step=None,
                                                          units="words", format="txt", keep_punct=False,
                                                          keep_sym=False,
                                                          max_samples=None)
        # THEN
        self.assertEqual(len([text for text in results if text["lang"] != 'NA']), 5)

        # WHEN
        results = superstyl.preproc.pipe.docs_to_samples(self.paths, identify_lang=False, size=2, step=None,
                                                         units="words", format="txt", keep_punct=False,
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
    # First, testing the pipe features
    def test_normalise(self):
        text = " Hello,  Mr. 𓀁, how are §§ you; doing?"
        expected_default = "hello mr how are you doing"
        self.assertEqual(superstyl.preproc.pipe.normalise(text), expected_default)
        expected_keeppunct = "Hello, Mr. , how are SSSS you; doing?"
        self.assertEqual(superstyl.preproc.pipe.normalise(text, keep_punct=True), expected_keeppunct)
        expected_keepsym = "Hello, Mr. 𓀁, how are §§ you; doing?" #TODO: modify test according to new def
        self.assertEqual(superstyl.preproc.pipe.normalise(text, keep_sym=True), expected_keepsym)

    def test_detect_lang(self):
        french = "Bonjour, Monsieur, comment allez-vous?"
        # NB: it fails on that !!!
        # english = "Hello, How do you do good sir?"
        # still too hard
        # english = "Hello, How do you do good sir? Are you well today?"
        english = "Hello, How do you do good sir? Are you well today? Is this so bloody hard? Really, this is still failing?"
        italian = "Buongiorno signore, come sta?"
        #TODO: find something that manages old languages, like fasttext did…
        self.assertEqual(superstyl.preproc.pipe.detect_lang(french), "fr")
        self.assertEqual(superstyl.preproc.pipe.detect_lang(english), "en")
        self.assertEqual(superstyl.preproc.pipe.detect_lang(italian), "it")

    # Now, lower level features,
    # from features_extract
    def test_counts(self):
        # Scenario: given a text, extract a list of the features that appear in it, with their counts in absolute frequency
        # GIVEN
        text = "the cat the dog the squirrel the cat the cat"
        # WHEN
        results = superstyl.preproc.features_extract.count_features(text, feats ="words", n = 1)
        # THEN
        expected = ({'the': 5, 'cat': 3, 'dog': 1, 'squirrel': 1}, 10)
        self.assertEqual(results, expected)

        # WHEN
        results = superstyl.preproc.features_extract.count_features(text, feats="words", n=2)
        # THEN
        expected = ({'the_cat': 3, 'cat_the': 2, 'the_dog': 1, 'dog_the': 1, 'the_squirrel': 1, 'squirrel_the': 1}, 9)
        self.assertEqual(results, expected)

        # GIVEN
        text = "These yo yo!"
        # WHEN
        results = superstyl.preproc.features_extract.count_features(text, feats="chars", n=3)
        # THEN
        expected = ({'_yo': 2, 'The': 1, 'hes': 1, 'ese': 1, 'se_': 1, 'e_y': 1, 'yo_': 1, 'o_y': 1, 'yo!': 1}, 10)
        self.assertEqual(results, expected)

        # GIVEN
        text = "These yo yo!"
        # WHEN
        results = superstyl.preproc.features_extract.count_features(text, feats="affixes", n=3)
        # THEN
        expected = ({'_yo': 2, 'The': 1, 'ese': 1, 'se_': 1, 'yo_': 1, 'yo!': 1}, 10)
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
        results = superstyl.preproc.pipe.max_sampling(myTexts, max_samples=1)
        # EXPECT
        self.assertEqual(len([text for text in results if text["aut"] == 'Smith']), 1)


# TODO: tests for SVM, etc.
# Test all options of main commands, see if they are accepted or not

if __name__ == '__main__':
    unittest.main()
