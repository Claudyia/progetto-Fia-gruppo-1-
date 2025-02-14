import unittest
import pandas as pd
from DataPreprocessing import PreprocessingDataset

class TestPreprocessingDataset(unittest.TestCase):

    def setUp(self):
        """Setup iniziale: crea un'istanza della classe e un dataset di test."""
        self.processor = PreprocessingDataset()
        
        # Creiamo un dataset di test con tutte le colonne richieste
        data = {
            "clump_thickness_ty": [1, 2, None, 4, 5],
            "uniformity_cellsize_xx": [1, 2, 3, 4, 5],
            "Uniformity of Cell Shape": [5, None, 3, 2, 1],
            "Marginal Adhesion": [1, 0, 1, 0, None],
            "Single Epithelial Cell Size": [2, 2, None, 3, 3],
            "bareNucleix_wrong": [4, 4, 4, None, 4],
            "Bland Chromatin": [3, 3, 3, 3, None],
            "Normal Nucleoli": [1, 2, 3, 4, None],
            "Mitoses": [0, 1, 0, None, 1],
            "classtype_v1": [0, 1, 0, 1, None]  # Target con un valore NaN per testare la pulizia
        }

        self.dataset = pd.DataFrame(data)
        self.processor.dataset = self.dataset

    def test_pulisci_dataset(self):
        """Testa la pulizia del dataset."""
        cleaned_data, _ = self.processor.pulisci_dataset()
        self.assertIsNotNone(cleaned_data)
        self.assertFalse(cleaned_data.isnull().values.any(), "Il dataset non dovrebbe avere valori NaN dopo la pulizia.")

    def test_separa_features_e_target(self):
        """Testa la separazione delle features e del target."""
        features, target = self.processor.separa_features_e_target(self.dataset)
        self.assertIsNotNone(features)
        self.assertIsNotNone(target)
        self.assertEqual(features.shape[1], 9, "Il numero di colonne delle features è errato.")
        self.assertEqual(target.shape[1], 1, "Il numero di colonne del target è errato.")

    def test_normalizza_features(self):
        """Testa la normalizzazione delle features."""
        features, _ = self.processor.pulisci_dataset()
        normalized_features = self.processor.normalizza_features()
        self.assertIsNotNone(normalized_features)
        self.assertEqual(features.shape, normalized_features.shape, "Il dataset normalizzato dovrebbe avere la stessa forma del dataset originale.")

if __name__ == '__main__':
    unittest.main()
    