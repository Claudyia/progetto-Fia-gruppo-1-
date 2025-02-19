import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ValidationStrategies import ValidationFactory, HoldOut, KfoldCrossValidationKNN, RandomSubsampling
from knn import KNN

class TestValidationStrategies(unittest.TestCase):

    def setUp(self):
        """
        Inizializza i dati per i test.
        """
        self.X = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100)
        })
        self.y = pd.DataFrame({
            'target': np.random.randint(2, size=100)
        })
        self.knn = KNN(k=3)  # Creiamo un'istanza della classe KNN

    def test_holdout_split(self):
        """
        Testa il metodo split della classe HoldOut.
        """
        holdout = HoldOut(dim_test_set=0.2)
        x_train, x_test, y_train, y_test = holdout.split(self.X, self.y)
        self.assertEqual(len(x_train), 80)
        self.assertEqual(len(x_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)

    def test_holdout_evaluate(self):
        """
        Testa il metodo evaluate della classe HoldOut.
        """
        holdout = HoldOut(dim_test_set=0.2)
        for metrica in range(1, 8):
            with self.subTest(metrica=metrica):
                result, confusion_matrix = holdout.evaluate(self.knn, self.X, self.y, metrica=metrica)
                self.assertIsNotNone(result)
                self.assertEqual(confusion_matrix.shape, (2, 2))

    def test_kfold_split(self):
        """ Testa il metodo split della classe KfoldCrossValidationKNN."""
        kfold = KfoldCrossValidationKNN(K=5)
        folds = kfold.split(self.X, self.y)
        self.assertEqual(len(folds), 5)
        for train_indices, test_indices in folds:
            self.assertEqual(len(train_indices) + len(test_indices), 100)

    def test_kfold_evaluate(self):
        """ Testa il metodo evaluate della classe KfoldCrossValidationKNN."""
        kfold = KfoldCrossValidationKNN(K=5)
        for metrica in range(1, 8):
            with self.subTest(metrica=metrica):
                result, confusion_matrices = kfold.evaluate(self.knn, self.X, self.y, metrica=metrica)
                self.assertIsNotNone(result)
                self.assertEqual(len(confusion_matrices), 5)
                for cm in confusion_matrices:
                    self.assertEqual(cm.shape, (2, 2))

    def test_random_subsampling_split(self):
        """ Testa il metodo split della classe RandomSubsampling."""
        random_subsampling = RandomSubsampling(K=5, dim_test_set=0.2)
        results = random_subsampling.split(self.X, self.y)
        self.assertEqual(len(results), 5)
        for X_train, y_train, X_test, y_test in results:
            self.assertEqual(len(X_train), 80)
            self.assertEqual(len(X_test), 20)
            self.assertEqual(len(y_train), 80)
            self.assertEqual(len(y_test), 20)

    def test_random_subsampling_evaluate(self):
        """ Testa il metodo evaluate della classe RandomSubsampling."""
        random_subsampling = RandomSubsampling(K=5, dim_test_set=0.2)
        for metrica in range(1, 8):
            with self.subTest(metrica=metrica):
                result, confusion_matrices = random_subsampling.evaluate(self.knn, self.X, self.y, metrica=metrica)
                self.assertIsNotNone(result)
                self.assertEqual(len(confusion_matrices), 5)
                for cm in confusion_matrices:
                    self.assertEqual(cm.shape, (2, 2))

    def test_validation_factory_holdout(self):
        """ Testa la creazione del metodo HoldOut tramite ValidationFactory."""
        validation_method = ValidationFactory.create_validation_method(1, dim_test_set=0.2)
        self.assertIsInstance(validation_method, HoldOut)

    def test_validation_factory_kfold(self):
        """ Testa la creazione del metodo KfoldCrossValidationKNN tramite ValidationFactory."""
        validation_method = ValidationFactory.create_validation_method(2, K=5)
        self.assertIsInstance(validation_method, KfoldCrossValidationKNN)

    def test_validation_factory_random_subsampling(self):
        """ Testa la creazione del metodo RandomSubsampling tramite ValidationFactory."""
        validation_method = ValidationFactory.create_validation_method(3, K=5, dim_test_set=0.2)
        self.assertIsInstance(validation_method, RandomSubsampling)

    def test_validation_factory_invalid_method(self):
        """ Testa la gestione di un metodo di validazione non valido tramite ValidationFactory."""
        with self.assertRaises(ValueError):
            ValidationFactory.create_validation_method(4)

if __name__ == '__main__':
    unittest.main()