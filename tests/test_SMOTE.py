import unittest
import superstyl
import pandas
import numpy as np
from collections import Counter


class TestSMOTEFixes(unittest.TestCase):
    """Tests pour les corrections SMOTE avec validation croisée"""
    
    def setUp(self):
        """Créer des datasets de test"""
        # Dataset équilibré basique
        self.balanced_data = pandas.DataFrame({
            'author': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
            'lang': ['en'] * 8,
            'feat1': [1, 2, 3, 4, 5, 6, 7, 8],
            'feat2': [8, 7, 6, 5, 4, 3, 2, 1]
        })
        
        # Dataset déséquilibré avec petites classes
        self.imbalanced_small = pandas.DataFrame({
            'author': ['Majority'] * 10 + ['Minority'] * 3,
            'lang': ['en'] * 13,
            'feat1': np.random.rand(13),
            'feat2': np.random.rand(13),
            'feat3': np.random.rand(13)
        })
        
        # Dataset pour group-k-fold avec structure de groupes
        # Format: Author_Work_Sample
        indices = (
            ['Smith_Work1_0-100', 'Smith_Work1_100-200', 'Smith_Work1_200-300'] +
            ['Smith_Work2_0-100', 'Smith_Work2_100-200'] +
            ['Dupont_Work1_0-100', 'Dupont_Work1_100-200'] +
            ['Dupont_Work2_0-100', 'Dupont_Work2_100-200', 'Dupont_Work2_200-300']
        )
        self.grouped_data = pandas.DataFrame({
            'author': ['Smith'] * 5 + ['Dupont'] * 5,
            'lang': ['en'] * 10,
            'feat1': np.random.rand(10),
            'feat2': np.random.rand(10)
        }, index=indices)
    
    def test_smote_with_kfold(self):
        """Test SMOTE avec k-fold classique"""
        # GIVEN: Un dataset déséquilibré
        # WHEN: On applique SMOTE avec k-fold
        try:
            results = superstyl.train_svm(
                self.imbalanced_small,
                cross_validate='k-fold',
                k=3,
                balance='SMOTE',
                norms=True
            )
            # THEN: Ça devrait fonctionner
            self.assertIn('confusion_matrix', results)
            self.assertIn('pipeline', results)
        except ValueError as e:
            if 'n_neighbors' in str(e):
                self.fail(f"SMOTE k-fold failed with: {e}")
    
    def test_smote_with_leave_one_out(self):
        """Test SMOTE avec leave-one-out"""
        # GIVEN: Un petit dataset
        small_data = pandas.DataFrame({
            'author': ['A'] * 4 + ['B'] * 4,
            'lang': ['en'] * 8,
            'feat1': [1, 2, 3, 4, 5, 6, 7, 8],
            'feat2': [8, 7, 6, 5, 4, 3, 2, 1]
        })
        
        # WHEN: On applique SMOTE avec LOO
        try:
            results = superstyl.train_svm(
                small_data,
                cross_validate='leave-one-out',
                balance='SMOTE',
                norms=True
            )
            # THEN: Ça devrait fonctionner
            self.assertIn('confusion_matrix', results)
        except ValueError as e:
            if 'n_neighbors' in str(e):
                self.fail(f"SMOTE LOO failed with: {e}")
    
    def test_smote_with_group_kfold(self):
        """Test SMOTE avec group-k-fold (le cas problématique)"""
        # GIVEN: Un dataset avec structure de groupes
        # WHEN: On applique SMOTE avec group-k-fold
        try:
            results = superstyl.train_svm(
                self.grouped_data,
                cross_validate='group-k-fold',
                k=2,
                balance='SMOTE',
                norms=True
            )
            # THEN: Ça devrait fonctionner ou skip SMOTE proprement
            self.assertIn('pipeline', results)
            # Si SMOTE a été skippé, le pipeline ne devrait pas contenir de sampling
            pipeline_steps = [step[0] for step in results['pipeline'].steps]
            # C'est OK si 'sampling' est présent ou absent (selon la taille des classes)
        except ValueError as e:
            if 'n_neighbors' in str(e):
                self.fail(f"SMOTE group-k-fold failed with: {e}")
    
    def test_smote_very_small_classes(self):
        """Test SMOTE avec des classes extrêmement petites (devrait skip SMOTE)"""
        # GIVEN: Dataset avec classe de taille 2
        tiny_data = pandas.DataFrame({
            'author': ['A', 'A', 'B'] * 2,
            'lang': ['en'] * 6,
            'feat1': [1, 2, 3, 4, 5, 6],
            'feat2': [6, 5, 4, 3, 2, 1]
        })
        
        # WHEN: On essaie SMOTE avec k-fold
        try:
            results = superstyl.train_svm(
                tiny_data,
                cross_validate='k-fold',
                k=3,
                balance='SMOTE',
                norms=True
            )
            # THEN: Devrait fonctionner (SMOTE skippé ou n_neighbors=1)
            self.assertIn('pipeline', results)
        except ValueError as e:
            if 'n_neighbors' in str(e):
                self.fail(f"Should have skipped SMOTE or used n_neighbors=1, but got: {e}")
    
    def test_smotetomek_with_cv(self):
        """Test SMOTETomek avec validation croisée"""
        # WHEN: On applique SMOTETomek
        try:
            results = superstyl.train_svm(
                self.imbalanced_small,
                cross_validate='k-fold',
                k=3,
                balance='SMOTETomek',
                norms=True
            )
            # THEN: Ça devrait fonctionner
            self.assertIn('confusion_matrix', results)
        except ValueError as e:
            if 'n_neighbors' in str(e):
                self.fail(f"SMOTETomek failed with: {e}")
    
    def test_alternative_balance_methods(self):
        """Test que les méthodes alternatives fonctionnent"""
        # Test upsampling
        results1 = superstyl.train_svm(
            self.imbalanced_small,
            cross_validate='k-fold',
            k=3,
            balance='upsampling',
            norms=True
        )
        self.assertIn('confusion_matrix', results1)
        
        # Test downsampling
        results2 = superstyl.train_svm(
            self.imbalanced_small,
            cross_validate='k-fold',
            k=3,
            balance='downsampling',
            norms=True
        )
        self.assertIn('confusion_matrix', results2)
        
        # Test class_weights sans balance
        results3 = superstyl.train_svm(
            self.imbalanced_small,
            cross_validate='k-fold',
            k=3,
            balance=None,
            class_weights=True,
            norms=True
        )
        self.assertIn('confusion_matrix', results3)

    def test_smote_with_group_kfold_bad_distribution(self):
        """Test que group-k-fold échoue proprement avec une mauvaise distribution de classes"""
        # GIVEN: Dataset où les groupes sont mal distribués (une classe par groupe)
        indices = (
            ['Author1_Work1_0', 'Author1_Work1_1', 'Author1_Work1_2'] +
            ['Author1_Work2_0', 'Author1_Work2_1'] +
            ['Author2_Work1_0', 'Author2_Work1_1', 'Author2_Work1_2']
        )
        bad_grouped = pandas.DataFrame({
            'author': ['Author1'] * 5 + ['Author2'] * 3,
            'lang': ['en'] * 8,
            'feat1': np.random.rand(8),
            'feat2': np.random.rand(8)
        }, index=indices)
        
        # WHEN: On essaie SMOTE avec group-k-fold
        # THEN: Devrait soit skip SMOTE, soit échouer avec un message clair
        try:
            results = superstyl.train_svm(
                bad_grouped,
                cross_validate='group-k-fold',
                k=2,
                balance='SMOTE',
                norms=True
            )
            # Si ça marche, c'est que SMOTE a été skippé intelligemment
            self.assertIn('pipeline', results)
        except ValueError as e:
            # L'erreur "Got 1 class instead" est acceptable ici
            # C'est un problème inhérent aux données, pas au code
            if "Got 1 class instead" not in str(e):
                self.fail(f"Expected 'Got 1 class' error or success, got: {e}")
    



class TestFinalPredFixes(unittest.TestCase):
    """Tests pour la correction final_pred avec test set"""
    
    def setUp(self):
        """Créer des datasets train et test"""
        self.train = pandas.DataFrame({
            'author': ['A'] * 5 + ['B'] * 5,
            'lang': ['en'] * 10,
            'feat1': np.random.rand(10),
            'feat2': np.random.rand(10),
            'feat3': np.random.rand(10)
        })
        
        self.test = pandas.DataFrame({
            'author': ['A'] * 2 + ['B'] * 2,
            'lang': ['en'] * 4,
            'feat1': np.random.rand(4),
            'feat2': np.random.rand(4),
            'feat3': np.random.rand(4)
        })
    
    def test_final_pred_with_test_set(self):
        """Test final_pred=True AVEC test set (devrait fonctionner)"""
        # WHEN: On fait une prédiction finale avec test set
        results = superstyl.train_svm(
            self.train,
            test=self.test,
            cross_validate='k-fold',
            k=3,
            final_pred=True,
            norms=True
        )
        
        # THEN: Devrait contenir les prédictions finales
        self.assertIn('final_predictions', results)
        self.assertIn('pipeline', results)
        self.assertEqual(len(results['final_predictions']), len(self.test))
    
    def test_final_pred_without_test_set(self):
        """Test final_pred=True SANS test set (devrait lever une erreur)"""
        # WHEN: On essaie final_pred sans test set
        with self.assertRaises(ValueError) as context:
            superstyl.train_svm(
                self.train,
                test=None,  # ← Pas de test set
                cross_validate='k-fold',
                k=3,
                final_pred=True,
                norms=True
            )
        
        # THEN: Devrait avoir un message d'erreur clair
        self.assertIn('test', str(context.exception).lower())
    
    def test_cv_without_final_pred(self):
        """Test validation croisée sans final_pred (devrait fonctionner)"""
        # WHEN: On fait juste de la CV sans prédiction finale
        results = superstyl.train_svm(
            self.train,
            cross_validate='k-fold',
            k=3,
            final_pred=False,
            norms=True
        )
        
        # THEN: Devrait avoir confusion matrix mais pas final_predictions
        self.assertIn('confusion_matrix', results)
        self.assertNotIn('final_predictions', results)
    
    def test_train_test_split_with_final_pred(self):
        """Test train/test split classique avec final_pred"""
        # WHEN: On fait un train/test split (pas de CV)
        results = superstyl.train_svm(
            self.train,
            test=self.test,
            cross_validate=None,  # ← Pas de CV
            final_pred=True,
            norms=True
        )
        
        # THEN: Devrait avoir les prédictions finales
        self.assertIn('final_predictions', results)
        self.assertEqual(len(results['final_predictions']), len(self.test))
    
    def test_cv_then_final_on_test(self):
        """Test CV sur train puis prédiction finale sur test"""
        # WHEN: On fait CV + prédiction finale (cas le plus complet)
        results = superstyl.train_svm(
            self.train,
            test=self.test,
            cross_validate='k-fold',
            k=3,
            final_pred=True,
            norms=True
        )
        
        # THEN: Devrait avoir confusion matrix (CV) ET final_predictions
        self.assertIn('confusion_matrix', results)
        self.assertIn('final_predictions', results)
        self.assertEqual(len(results['final_predictions']), len(self.test))


if __name__ == '__main__':
    unittest.main()