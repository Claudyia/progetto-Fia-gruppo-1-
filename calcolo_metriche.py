from abc import ABC, abstractmethod
import numpy as np
from knn import KNN

class MetricheFactory:
    """Classe factory che crea le metriche richieste dall'utente."""
    @staticmethod
    def create_metriche(tipo_metrica):
        if tipo_metrica == 1:
            return Accuracy()
        elif tipo_metrica == 2:
            return Error()
        elif tipo_metrica == 3:
            return Sensitivity()
        elif tipo_metrica == 4:
            return Specificity()
        elif tipo_metrica == 5:
            return GeometricMean()
        elif tipo_metrica == 6:
            return AUC()
        elif tipo_metrica == 7:
            return AllMetrics()
        else:
            raise ValueError("Tipo metrica non valido!")

class Metriche(ABC):
    """Classe astratta che rappresenta le metriche da calcolare."""
    @abstractmethod
    def calcola(self, y_test, y_pred, **kwargs):
        pass

class Accuracy(Metriche):
    def calcola(self, y_test, y_pred):
        """Metodo che calcola l'accuratezza."""
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        return accuracy
    
class Error(Metriche):
    def calcola(self, y_test, y_pred):
        """Metodo che calcola l'errore."""
        error = np.sum(y_pred != y_test) / len(y_test)
        return error
    
class Sensitivity(Metriche):
    def calcola(self, y_test, y_pred):
        """Metodo che calcola la sensitivity."""
        TP = np.sum((y_pred == 2) & (y_test == 2))
        FN = np.sum((y_pred == 4) & (y_test == 2))
        sensitivity = TP/ (TP + FN)
        return sensitivity

    
class Specificity(Metriche):
    def calcola(self, y_test, y_pred):
        """Metodo che calcola la specificità."""
        TN = np.sum((y_pred == 4) & (y_test == 4))
        FP = np.sum((y_pred == 2) & (y_test == 4))
        specificity = TN / (TN + FP)
        return specificity

class GeometricMean(Metriche):
    def calcola(self, y_test, y_pred):
        """Metodo che calcola la media geometrica tra sensitivity e specificity."""
        sensitivity = Sensitivity().calcola(y_test, y_pred)
        specificity = Specificity().calcola(y_test, y_pred)
        geometric_mean = np.sqrt(sensitivity * specificity) if sensitivity * specificity> 0 else 0
        return geometric_mean
 
class AUC(Metriche):
    def calcola(self, y_test, y_scores, knn: KNN):
        """Metodo che calcola l'Area Under the Curve (AUC) della curva ROC, e disegna quest'ultima."""
        tprs, fprs = knn.ROC_curve(y_test, y_scores)
        auc = np.trapz(tprs, fprs)
        return auc, (tprs, fprs)
    
            
class AllMetrics(Metriche):
    def calcola(self, y_test, y_pred, y_scores, knn: KNN):
        """Metodo che calcola tutte le metriche e disegna la curva ROC."""
        accuracy = Accuracy().calcola(y_test, y_pred)
        error = Error().calcola(y_test, y_pred)
        sensitivity = Sensitivity().calcola(y_test, y_pred)
        specificity = Specificity().calcola(y_test, y_pred)
        geometric_mean = GeometricMean().calcola(y_test, y_pred)
        auc, roc_data = AUC().calcola(y_test, y_scores, knn=knn) 
        result = np.array([accuracy, error, sensitivity, specificity, geometric_mean, auc])
        return result, roc_data
    
