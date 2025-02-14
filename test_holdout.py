import unittest
import numpy as np
import pandas as pd
from HoldOut import HoldOut

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
        self.k = 3
        self.holdout = HoldOut(self.k)

    def test_holdout_split(self):
        x_train, x_test, y_train, y_test = self.holdout.Holdout_split(self.X, self.y, dim_test_set=0.2)
        
        # Verifica che la dimensione del test set sia corretta
        self.assertEqual(len(x_test), 20)
        self.assertEqual(len(y_test), 20)
        
        # Verifica che la dimensione del training set sia corretta
        self.assertEqual(len(x_train), 80)
        self.assertEqual(len(y_train), 80)
        
        # Verifica che non ci siano sovrapposizioni tra training e test set
        self.assertTrue(x_train.index.isin(x_test.index).sum() == 0)
        self.assertTrue(y_train.index.isin(y_test.index).sum() == 0)

    def test_evaluate(self):
        accuracy, error, sensitivity, specificity, geometric_mean, confusion_matrix = self.holdout.evaluate(self.X, self.y)
        
        # Verifica che i risultati siano nel range corretto
        self.assertTrue(0 <= accuracy <= 1)
        self.assertTrue(0 <= error <= 1)
        self.assertTrue(0 <= sensitivity <= 1)
        self.assertTrue(0 <= specificity <= 1)
        self.assertTrue(0 <= geometric_mean <= 1)
        
        # Verifica che la matrice di confusione abbia la dimensione corretta
        self.assertEqual(confusion_matrix.shape, (2, 2))

if __name__ == '__main__':
    unittest.main()