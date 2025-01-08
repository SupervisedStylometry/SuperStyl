import unittest
import superstyl
import os
import pandas
import tempfile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class Main_svm(unittest.TestCase):
    def test_train_svm(self):
        # SCENARIO: train a svm and test it
        # GIVEN
        train = pandas.DataFrame({'author': {'Dupont_Letter1.txt': 'Dupont', 'Smith_Letter1.txt': 'Smith', 'Smith_Letter2.txt': 'Smith'},
                           'lang': {'Dupont_Letter1.txt': 'NA', 'Smith_Letter1.txt': 'NA', 'Smith_Letter2.txt': 'NA'},
                           'this': {'Dupont_Letter1.txt': 0.0, 'Smith_Letter1.txt': 0.25, 'Smith_Letter2.txt': 0.2},
                           'is': {'Dupont_Letter1.txt': 0.0, 'Smith_Letter1.txt': 0.25, 'Smith_Letter2.txt': 0.2},
                           'the': {'Dupont_Letter1.txt': 0.0, 'Smith_Letter1.txt': 0.25, 'Smith_Letter2.txt': 0.2},
                           'text': {'Dupont_Letter1.txt': 0.0, 'Smith_Letter1.txt': 0.25, 'Smith_Letter2.txt': 0.2},
                           'voici': {'Dupont_Letter1.txt': 1/3, 'Smith_Letter1.txt': 0.0, 'Smith_Letter2.txt': 0.0},
                           'le': {'Dupont_Letter1.txt': 1/3, 'Smith_Letter1.txt': 0.0, 'Smith_Letter2.txt': 0.0},
                           'texte': {'Dupont_Letter1.txt': 1/3, 'Smith_Letter1.txt': 0.0, 'Smith_Letter2.txt': 0.0},
                           'also': {'Dupont_Letter1.txt': 0.0, 'Smith_Letter1.txt': 0.0, 'Smith_Letter2.txt': 0.2}})
        test = train # easier to make no mistake :)
        # WHEN
        results = superstyl.train_svm(train, test, final_pred=True)
        # THEN
        expected_results = {'filename': {0: 'Dupont_Letter1.txt', 1: 'Smith_Letter1.txt', 2: 'Smith_Letter2.txt'},
                            'author': {0: 'Dupont', 1: 'Smith', 2: 'Smith'},
                            'Decision function': {0: -0.7840855202465706, 1: 0.869169888531951, 2: 0.8619190699737995}}
        expected_keys = ['final_predictions', 'pipeline']
        self.assertEqual(results["final_predictions"].to_dict()['author'], expected_results['author'])
        self.assertEqual(list(results.keys()), expected_keys)
        # This is only the first minimal test for this function

        # WHEN
        results = superstyl.train_svm(train, test, final_pred=False)
        # THEN
        expected_results = \
            {"confusion_matrix": {'pred:Dupont': {'true:Dupont': 1, 'true:Smith': 0},
                                                   'pred:Smith': {'true:Dupont': 0, 'true:Smith': 2}},
             "classification_report": "              precision    recall  f1-score   support\n\n"
                                      "      Dupont       1.00      1.00      1.00         1\n"
                                      "       Smith       1.00      1.00      1.00         2\n\n"
                                      "    accuracy                           1.00         3\n"
                                      "   macro avg       1.00      1.00      1.00         3\n"
                                      "weighted avg       1.00      1.00      1.00         3\n",
             "misattributions": {'True': {}, 'Pred': {}}
             }
        #TODO: plus the sklearn pipeline

        expected_keys = ['confusion_matrix', 'classification_report', 'misattributions', 'pipeline']
        self.assertEqual(results["confusion_matrix"].to_dict(), expected_results["confusion_matrix"])
        self.assertEqual(results["classification_report"], expected_results["classification_report"])
        self.assertEqual(results["misattributions"].to_dict(), expected_results["misattributions"])
        self.assertEqual(list(results.keys()), expected_keys)

        #TODO: quick tests for SMOTE, SMOTETOMEK, to improve
        # WHEN
        results = superstyl.train_svm(train, test, final_pred=False, balance="SMOTETomek")
        # THEN
        self.assertEqual(results["confusion_matrix"].to_dict(), expected_results["confusion_matrix"])
        self.assertEqual(results["classification_report"], expected_results["classification_report"])
        self.assertEqual(results["misattributions"].to_dict(), expected_results["misattributions"])
        self.assertEqual(list(results.keys()), expected_keys)

        # WHEN
        results = superstyl.train_svm(train, test, final_pred=False, balance="SMOTE")
        # THEN
        self.assertEqual(results["confusion_matrix"].to_dict(), expected_results["confusion_matrix"])
        self.assertEqual(results["classification_report"], expected_results["classification_report"])
        self.assertEqual(results["misattributions"].to_dict(), expected_results["misattributions"])
        self.assertEqual(list(results.keys()), expected_keys)

        # now, when it can be applied, but needs to be recomputed, because of a maximum possible number of
        # neighbors < 5
        train2 = pandas.DataFrame({'author': {'Dupont_Letter1.txt': 'Dupont', 'Dupont_Letter2.txt': 'Dupont', 'Smith_Letter1.txt': 'Smith', 'Smith_Letter2.txt': 'Smith'},
                           'lang': {'Dupont_Letter1.txt': 'NA', 'Dupont_Letter2.txt': 'NA', 'Smith_Letter1.txt': 'NA', 'Smith_Letter2.txt': 'NA'},
                           'this': {'Dupont_Letter1.txt': 0.0, 'Dupont_Letter2.txt': 0.0, 'Smith_Letter1.txt': 0.25, 'Smith_Letter2.txt': 0.2},
                           'is': {'Dupont_Letter1.txt': 0.0, 'Dupont_Letter2.txt': 0.0, 'Smith_Letter1.txt': 0.25, 'Smith_Letter2.txt': 0.2},
                           'the': {'Dupont_Letter1.txt': 0.0, 'Dupont_Letter2.txt': 0.0, 'Smith_Letter1.txt': 0.25, 'Smith_Letter2.txt': 0.2},
                           'text': {'Dupont_Letter1.txt': 0.0, 'Dupont_Letter2.txt': 0.0, 'Smith_Letter1.txt': 0.25, 'Smith_Letter2.txt': 0.2},
                           'voici': {'Dupont_Letter1.txt': 1/3, 'Dupont_Letter2.txt': 1/3, 'Smith_Letter1.txt': 0.0, 'Smith_Letter2.txt': 0.0},
                           'le': {'Dupont_Letter1.txt': 1/3, 'Dupont_Letter2.txt': 1/3, 'Smith_Letter1.txt': 0.0, 'Smith_Letter2.txt': 0.0},
                           'texte': {'Dupont_Letter1.txt': 1/3, 'Dupont_Letter2.txt': 1/3, 'Smith_Letter1.txt': 0.0, 'Smith_Letter2.txt': 0.0},
                           'also': {'Dupont_Letter1.txt': 0.0, 'Dupont_Letter2.txt': 0.0, 'Smith_Letter1.txt': 0.0, 'Smith_Letter2.txt': 0.2}})

        # WHEN
        results = superstyl.train_svm(train2, test, final_pred=True, balance="SMOTETomek")
        # THEN
        expected_preds = {'filename': {0: 'Dupont_Letter1.txt', 1: 'Smith_Letter1.txt',
                                                        2: 'Smith_Letter2.txt'},
                                           'author': {0: 'Dupont', 1: 'Smith', 2: 'Smith'},
                                           'Decision function': {0: -0.883772535448984, 1: 0.8756912342726781,
                                                                 2: 0.873288374519472}}

        self.assertEqual(results['final_predictions'].to_dict()["author"], expected_preds["author"])

        # WHEN
        results = superstyl.train_svm(train2, test, final_pred=True, balance="SMOTE")
        # THEN
        self.assertEqual(results['final_predictions'].to_dict()["author"], expected_preds["author"])


    def test_plot_rolling(self):
        train = pandas.DataFrame({
            'author': {'Text_0-1000': 'A', 'Text_1000-2000': 'A', 'Text_2000-3000': 'B'},
            'lang': {'Text_0-1000': 'NA', 'Text_1000-2000': 'NA', 'Text_2000-3000': 'NA'},
            'word1': {'Text_0-1000': 0.5, 'Text_1000-2000': 0.5, 'Text_2000-3000': 0.0},
            'word2': {'Text_0-1000': 0.0, 'Text_1000-2000': 0.5, 'Text_2000-3000': 0.5},
            'word3': {'Text_0-1000': 0.5, 'Text_1000-2000': 0.0, 'Text_2000-3000': 0.5}
        })
        test = train.copy()
        results = superstyl.train_svm(train, test, final_pred=True)

        try:
            superstyl.svm.plot_rolling(results["final_predictions"], smoothing=3)
        except Exception as e:
            self.fail(f"plot_rolling_stylometry raised an exception: {e}")

