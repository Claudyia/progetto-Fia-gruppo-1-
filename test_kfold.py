import unittest
import numpy as np
import pandas as pd
from Kfold_CrossValidation import KfoldCrossValidationKNN

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
        self.k = 3
        self.K = 5
        self.kfold = KfoldCrossValidationKNN(self.k, self.K)

    def test_kfold_split(self):
        folds = self.kfold.kfold_split(self.X)
        
        # Verifica che il numero di fold sia corretto
        self.assertEqual(len(folds), self.K)
        
        # Verifica che ogni fold abbia la dimensione corretta
        fold_size = len(self.X) // self.K
        for train_indices, test_indices in folds:
            self.assertEqual(len(test_indices), fold_size)
            self.assertEqual(len(train_indices), len(self.X) - fold_size)
            
            # Verifica che non ci siano sovrapposizioni tra training e test set
            self.assertTrue(np.intersect1d(train_indices, test_indices).size == 0)

    def test_evaluate(self):
        accuracy, error, sensitivity, specificity, geometric_mean, model_accuracy, model_error, model_sensitivity, model_specificity, model_geometric_mean, confusion_matrix = self.kfold.evaluate(self.X, self.y)
        
        # Verifica che i risultati siano nel range corretto
        self.assertTrue(0 <= model_accuracy <= 1)
        self.assertTrue(0 <= model_error <= 1)
        self.assertTrue(0 <= model_sensitivity <= 1)
        self.assertTrue(0 <= model_specificity <= 1)
        self.assertTrue(0 <= model_geometric_mean <= 1)
        
        # Verifica che la matrice di confusione abbia la dimensione corretta
        self.assertEqual(confusion_matrix.shape, (self.K, 2, 2))

if __name__ == '__main__':
    unittest.main()