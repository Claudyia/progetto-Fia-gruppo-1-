import unittest
import pandas as pd
import numpy as np
from ValidationStrategies import HoldOut, KfoldCrossValidationKNN, RandomSubsampling
from knn import KNN

class TestRandomSubsampling(unittest.TestCase):

    def setUp(self):
        # Creiamo un dataset di esempio per i test
        self.X = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.rand(100)
        })
        self.y = pd.DataFrame({
            'target': np.random.choice([2, 4], size=100)
        })
        self.K = 5
        self.dim_test_set = 0.2
        self.random_subsampling = RandomSubsampling(self.K, self.dim_test_set)

    def test_random_subsampling_split(self):
        risultati_iterazioni = self.random_subsampling.split(self.X, self.y)
        
        # Verifica che il numero di iterazioni sia corretto
        self.assertEqual(len(risultati_iterazioni), self.K)
        
        # Verifica che ogni iterazione abbia la dimensione corretta
        num_campioni_test = int(len(self.X) * self.dim_test_set)
        for X_train, y_train, X_test, y_test in risultati_iterazioni:
            self.assertEqual(len(X_test), num_campioni_test)
            self.assertEqual(len(y_test), num_campioni_test)
            self.assertEqual(len(X_train), len(self.X) - num_campioni_test)
            self.assertEqual(len(y_train), len(self.y) - num_campioni_test)
            
            # Verifica che non ci siano sovrapposizioni tra training e test set
            self.assertTrue(X_train.index.isin(X_test.index).sum() == 0)
            self.assertTrue(y_train.index.isin(y_test.index).sum() == 0)

    def test_evaluate(self):
        knn = KNN(k=3)
        accuracy_media, error_rate_media, sensitivity_media, specificity_media, geometric_mean_media, confusion_matrix_iterazione = self.random_subsampling.evaluate(knn, self.X, self.y)
        
        # Verifica che i risultati siano nel range corretto
        self.assertTrue(0 <= accuracy_media <= 1)
        self.assertTrue(0 <= error_rate_media <= 1)
        self.assertTrue(0 <= sensitivity_media <= 1)
        self.assertTrue(0 <= specificity_media <= 1)
        self.assertTrue(0 <= geometric_mean_media <= 1)
        
        # Verifica che la matrice di confusione abbia la dimensione corretta
        self.assertEqual(len(confusion_matrix_iterazione), self.K)
        for matrix in confusion_matrix_iterazione:
            self.assertEqual(matrix.shape, (2, 2))

class TestHoldOut(unittest.TestCase):
    
    def setUp(self):
        # Creiamo un dataset di esempio per i test
        self.X = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.rand(100)
        })
        self.y = pd.DataFrame({
            'target': np.random.choice([2, 4], size=100)
        })
        self.dim_test_set = 0.2
        self.holdout = HoldOut(self.dim_test_set)

    def test_holdout_split(self):
        x_train, x_test, y_train, y_test = self.holdout.split(self.X, self.y)
        
        # Verifica che la dimensione del test set sia corretta
        self.assertEqual(len(x_test), int(len(self.X) * self.dim_test_set))
        self.assertEqual(len(y_test), int(len(self.y) * self.dim_test_set))
        
        # Verifica che la dimensione del training set sia corretta
        self.assertEqual(len(x_train), len(self.X) - int(len(self.X) * self.dim_test_set))
        self.assertEqual(len(y_train), len(self.y) - int(len(self.y) * self.dim_test_set))
        
        # Verifica che non ci siano sovrapposizioni tra training e test set
        self.assertTrue(x_train.index.isin(x_test.index).sum() == 0)
        self.assertTrue(y_train.index.isin(y_test.index).sum() == 0)

    def test_evaluate(self):
        knn = KNN(k=3)
        accuracy, error, sensitivity, specificity, geometric_mean, confusion_matrix = self.holdout.evaluate(knn, self.X, self.y)
        
        # Verifica che i risultati siano nel range corretto
        self.assertTrue(0 <= accuracy <= 1)
        self.assertTrue(0 <= error <= 1)
        self.assertTrue(0 <= sensitivity <= 1)
        self.assertTrue(0 <= specificity <= 1)
        self.assertTrue(0 <= geometric_mean <= 1)
        
        # Verifica che la matrice di confusione abbia la dimensione corretta
        self.assertEqual(confusion_matrix.shape, (2, 2)) 

class TestKfoldCrossValidationKNN(unittest.TestCase):
    def setUp(self):
        # Creiamo un dataset di esempio per i test
        self.X = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.rand(100)
        })
        self.y = pd.DataFrame({
            'target': np.random.choice([2, 4], size=100)
        })
        self.K = 5
        self.kfold = KfoldCrossValidationKNN(self.K)

    def test_kfold_split(self):
        risultati_iterazioni = self.kfold.split(self.X)
        
        # Verifica che il numero di iterazioni sia corretto
        self.assertEqual(len(risultati_iterazioni), self.K)
        
        # Verifica che ogni iterazione abbia la dimensione corretta
        num_campioni_test = int(len(self.X) / self.K)
        for X_train, X_test, in risultati_iterazioni:
            self.assertEqual(len(X_test), num_campioni_test)
            self.assertEqual(len(X_train), len(self.X) - num_campioni_test)
            
            # Verifica che non ci siano sovrapposizioni tra training e test set
            self.assertTrue(np.intersect1d(X_train, X_test).size == 0)
  



    def test_evaluate(self):
        knn = KNN(k=3)
        accuracy_media, error_rate_media, sensitivity_media, specificity_media, geometric_mean_media, confusion_matrix_iterazione = self.kfold.evaluate(knn, self.X, self.y)
        
        # Verifica che i risultati siano nel range corretto
        self.assertTrue(0 <= accuracy_media <= 1)
        self.assertTrue(0 <= error_rate_media <= 1)
        self.assertTrue(0 <= sensitivity_media <= 1)
        self.assertTrue(0 <= specificity_media <= 1)
        # Verifica che il numero di iterazioni sia corretto
        risultati_iterazioni = self.kfold.split(self.X, self.y)
        self.assertEqual(len(risultati_iterazioni), self.K)

        # Verifica che ogni iterazione abbia la dimensione corretta
        num_campioni_test = int(len(self.X) / self.K)
        for X_train, y_train, X_test, y_test in risultati_iterazioni:
            self.assertEqual(len(X_test), num_campioni_test)
            self.assertEqual(len(y_test), num_campioni_test)
            self.assertEqual(len(X_train), len(self.X) - num_campioni_test)
            self.assertEqual(len(y_train), len(self.y) - num_campioni_test)
            
            # Verifica che non ci siano sovrapposizioni tra training e test set
            self.assertTrue(X_train.index.isin(X_test.index).sum() == 0)
            self.assertTrue(y_train.index.isin(y_test.index).sum() == 0)


    def test_evaluate(self):
        knn = KNN(k=3)
        accuracy_media, error_rate_media, sensitivity_media, specificity_media, geometric_mean_media, confusion_matrix_iterazione = self.kfold.evaluate(knn, self.X, self.y)
        
        # Verifica che i risultati siano nel range corretto
        self.assertTrue(0 <= accuracy_media <= 1)
        self.assertTrue(0 <= error_rate_media <= 1)
        self.assertTrue(0 <= sensitivity_media <= 1)
        self.assertTrue(0 <= specificity_media <= 1)
        self.assertTrue(0 <= geometric_mean_media <= 1)
        
        # Verifica che la matrice di confusione abbia la dimensione corretta
        self.assertEqual(len(confusion_matrix_iterazione), self.K)

        for matrix in confusion_matrix_iterazione:
            self.assertEqual(matrix.shape, (2, 2))

    

if __name__ == '__main__':
    unittest.main()