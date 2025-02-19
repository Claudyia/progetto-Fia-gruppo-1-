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
        expected_accuracy = np.sum(self.y_pred == self.y_test) / len(self.y_test)
        self.assertAlmostEqual(accuracy, expected_accuracy, places=4)

    def test_error(self):
        """Testa il calcolo dell'errore."""
        error = Error().calcola(self.y_test, self.y_pred)
        expected_error = np.sum(self.y_pred != self.y_test) / len(self.y_test)
        self.assertAlmostEqual(error, expected_error, places=4)

    def test_sensitivity(self):
        """Testa il calcolo della sensitivity."""
        sensitivity = Sensitivity().calcola(self.y_test, self.y_pred)
        TP = np.sum((self.y_pred == 2) & (self.y_test == 2))
        FN = np.sum((self.y_pred == 4) & (self.y_test == 2))
        expected_sensitivity = TP / (TP + FN)
        self.assertAlmostEqual(sensitivity, expected_sensitivity, places=4)

    def test_specificity(self):
        """Testa il calcolo della specificità."""
        specificity = Specificity().calcola(self.y_test, self.y_pred)
        TN = np.sum((self.y_pred == 4) & (self.y_test == 4))
        FP = np.sum((self.y_pred == 2) & (self.y_test == 4))
        expected_specificity = TN / (TN + FP)
        self.assertAlmostEqual(specificity, expected_specificity, places=4)

    def test_geometric_mean(self):
        """Testa il calcolo della media geometrica."""
        geometric_mean = GeometricMean().calcola(self.y_test, self.y_pred)
        sensitivity = Sensitivity().calcola(self.y_test, self.y_pred)
        specificity = Specificity().calcola(self.y_test, self.y_pred)
        expected_geometric_mean = np.sqrt(sensitivity * specificity) if sensitivity * specificity > 0 else 0
        self.assertAlmostEqual(geometric_mean, expected_geometric_mean, places=4)

    def test_auc(self):
        """Testa il calcolo dell'AUC."""
        auc = AUC().calcola(self.y_test, self.y_scores, self.knn)
        tprs, fprs = self.knn.ROC_curve(self.y_test, self.y_scores)
        expected_auc = np.trapz(tprs, fprs)
        self.assertAlmostEqual(auc, expected_auc, places=4)

    def test_all_metrics(self):
        """Testa il calcolo di tutte le metriche."""
        all_metrics = AllMetrics().calcola(self.y_test, self.y_pred, self.y_scores, self.knn)
        accuracy = Accuracy().calcola(self.y_test, self.y_pred)
        error = Error().calcola(self.y_test, self.y_pred)
        sensitivity = Sensitivity().calcola(self.y_test, self.y_pred)
        specificity = Specificity().calcola(self.y_test, self.y_pred)
        geometric_mean = GeometricMean().calcola(self.y_test, self.y_pred)
        auc = AUC().calcola(self.y_test, self.y_scores, self.knn)
        expected_all_metrics = np.array([accuracy, error, sensitivity, specificity, geometric_mean, auc])
        np.testing.assert_array_almost_equal(all_metrics, expected_all_metrics, decimal=4)

if __name__ == '__main__':
    unittest.main()