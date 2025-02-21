import unittest
import numpy as np
from calcolo_metriche import MetricheFactory, Accuracy, Error, Sensitivity, Specificity, GeometricMean, AUC, AllMetrics
from knn import KNN

class TestMetriche(unittest.TestCase):

    def setUp(self): 
        """Setup iniziale: crea un dataset di test."""
        self.y_test = np.array([2, 4, 2, 4, 2, 4, 2, 4])
        self.y_pred = np.array([2, 4, 2, 4, 4, 2, 2, 4])
        self.y_scores = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
        self.knn = KNN(k=3)

    def test_accuracy(self):
        """Testa il calcolo dell'accuratezza."""
        accuracy = Accuracy().calcola(self.y_test, self.y_pred)
        expected_accuracy = 0.75
        self.assertAlmostEqual(accuracy, expected_accuracy, places=4)

    def test_error(self):
        """Testa il calcolo dell'errore."""
        error = Error().calcola(self.y_test, self.y_pred)
        expected_error = 0.25
        self.assertAlmostEqual(error, expected_error, places=4)

    def test_sensitivity(self):
        """Testa il calcolo della sensitivity."""
        sensitivity = Sensitivity().calcola(self.y_test, self.y_pred)
        expected_sensitivity = 0.75
        self.assertAlmostEqual(sensitivity, expected_sensitivity, places=4)

    def test_specificity(self):
        """Testa il calcolo della specificità."""
        specificity = Specificity().calcola(self.y_test, self.y_pred)
        expected_specificity = 0.75
        self.assertAlmostEqual(specificity, expected_specificity, places=4)

    def test_geometric_mean(self):
        """Testa il calcolo della media geometrica."""
        geometric_mean = GeometricMean().calcola(self.y_test, self.y_pred)
        expected_geometric_mean = 0.75
        self.assertAlmostEqual(geometric_mean, expected_geometric_mean, places=4)

    def test_auc(self):
        """Testa il calcolo dell'AUC."""
        auc, roc_data = AUC().calcola(self.y_test, self.y_scores, self.knn)
        tprs_expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0])
        fprs_expected = np.array([0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0])
        expected_auc = np.trapz(tprs_expected, fprs_expected)
        expected_roc_data = (tprs_expected, fprs_expected)
        np.testing.assert_array_almost_equal(roc_data, expected_roc_data, decimal=4)
        self.assertAlmostEqual(auc, expected_auc, places=4)

    def test_all_metrics(self):
        """Testa il calcolo di tutte le metriche."""
        all_metrics, roc_data = AllMetrics().calcola(self.y_test, self.y_pred, self.y_scores, self.knn)
        accuracy = 0.75
        error = 0.25
        sensitivity = 0.75
        specificity = 0.75
        geometric_mean = 0.75
        tprs_expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0])
        fprs_expected = np.array([0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0])
        roc_expected = (tprs_expected, fprs_expected)
        auc = np.trapz(tprs_expected, fprs_expected)
        expected_all_metrics = np.array([accuracy, error, sensitivity, specificity, geometric_mean, auc])
        np.testing.assert_array_almost_equal(all_metrics, expected_all_metrics, decimal=4)
        np.testing.assert_array_almost_equal(roc_data, roc_expected, decimal=4)

if __name__ == '__main__':
    unittest.main()