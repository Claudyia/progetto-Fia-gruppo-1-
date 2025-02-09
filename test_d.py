import unittest
import pandas as pd
from DataPreprocessing import PreprocessingDataset


def test_pulisci_dataset(self):
        """Testa la pulizia del dataset."""
        cleaned_data = self.processor.pulisci_dataset()
        self.assertIsNotNone(cleaned_data)
        self.assertFalse(cleaned_data.isnull().values.any(), "Il dataset non dovrebbe avere valori NaN dopo la pulizia.")
    
def test_separa_features_e_target(self):
        """Testa la separazione delle features e del target."""
        x, y = self.processor.separa_features_e_target(self.dataset.fillna(0))
        self.assertEqual(x.shape[1], 9, "Le features devono avere 9 colonne.")
        self.assertEqual(y.shape[1], 1, "Il target deve avere 1 colonna.")
    
def test_normalizza_features(self):
        """Testa la normalizzazione delle features."""
        self.processor.x, self.processor.y = self.processor.separa_features_e_target(self.dataset.fillna(0))
        x_normalized = self.processor.normalizza_features()
        self.assertTrue(((x_normalized >= 0) & (x_normalized <= 1)).all().all(), "I valori devono essere normalizzati tra 0 e 1.")

if __name__ == '__main__':
    unittest.main()
