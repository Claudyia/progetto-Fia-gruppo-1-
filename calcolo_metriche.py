from abc import ABC, abstractmethod
import numpy as np
from knn import KNN

class MetricheFactory:
    """Questa classe implementa il design pattern Factory per selezionare
    dinamicamente le metriche di valutazione da utilizzare."""
    
    @staticmethod
    def create_metriche(tipo_metrica:int):
        """
        Questo metodo restituisce un'istanza della metrica corrispondente al valore selezionato.

        Parameters
        ----------
        tipo_metrica : int
            Numero intero che identifica la metrica:
            - 1 → Accuracy
            - 2 → Error
            - 3 → Sensitivity 
            - 4 → Specificity
            - 5 → Geometric Mean
            - 6 → AUC (Area Under Curve)
            - 7 → Tutte le metriche insieme

        Returns
        -------
            Istanza della classe corrispondente alla metrica selezionata.
        """
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
    """
    Classe astratta che rappresenta una metrica di valutazione.

    Ogni metrica deve implementare il metodo calcola(), che verrà definito
    nelle classi derivate.
    """
    @abstractmethod
    def calcola(self, y_test, y_pred, **kwargs):
        pass

# Implementazione delle singole metriche:
class Accuracy(Metriche):
    """
    Classe per calcolare l'Accuracy, ovvero la percentuale di predizioni corrette.
    """
    def calcola(self, y_test, y_pred) -> float:
        """
        Metodo che calcola l'accuratezza del modello.

        Parameters
        ----------
        y_test : np.ndarray
            Etichette reali del dataset di test.
        y_pred : np.ndarray
            Etichette predette dal modello.

        Returns
        -------
        float
            Accuracy, valore compreso tra 0 e 1.
        """
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        return accuracy
    
class Error(Metriche):
    """
    Classe che calcola l'Error Rate, ovvero la percentuale di predizioni errate.
    """
    def calcola(self, y_test, y_pred) -> float:
        """
        Questo metodo calcola la percentuale di errore del modello.

        Parameters
        ----------
        y_test : np.ndarray
            Etichette reali del dataset di test.
        y_pred : np.ndarray
            Etichette predette dal modello.

        Returns
        -------
        float
            Error rate, valore compreso tra 0 e 1.
        """
        error = np.sum(y_pred != y_test) / len(y_test)
        return error
    
class Sensitivity(Metriche):
    """
    Questa classe calcola la Sensitivity, che misura la capacità del modello 
    di identificare correttamente i casi positivi (True Positive Rate).
    """
    def calcola(self, y_test, y_pred) -> float:
        """
        Questo metodo calcola la Sensitivity.

        Parameters
        ----------
        y_test : np.ndarray
            Etichette reali del dataset di test.
        y_pred : np.ndarray
            Etichette predette dal modello.

        Returns
        -------
        float
            Sensitivity, valore compreso tra 0 e 1.
        """
        TP = np.sum((y_pred == 2) & (y_test == 2))
        FN = np.sum((y_pred == 4) & (y_test == 2))
        sensitivity = TP/ (TP + FN)
        return sensitivity

    
class Specificity(Metriche):
    """
    Questa classe calcola la Specificity, che misura la capacità del modello di identificare
    correttamente i casi negativi (True Negative Rate).
    """
    def calcola(self, y_test, y_pred) -> float:
        """
        Questo metodo calcola la Specificity.

        Parameters
        ----------
        y_test : np.ndarray
            Etichette reali del dataset di test.
        y_pred : np.ndarray
            Etichette predette dal modello.

        Returns
        -------
        float
            Specificity, valore compreso tra 0 e 1.
        """
        TN = np.sum((y_pred == 4) & (y_test == 4))
        FP = np.sum((y_pred == 2) & (y_test == 4))
        specificity = TN / (TN + FP)
        return specificity

class GeometricMean(Metriche):
    """
    Questa classe calcola la Geometric Mean, che combina Sensitivity e Specificity per fornire
    una misura bilanciata delle prestazioni del modello.
    """
    def calcola(self, y_test, y_pred):
        """
        Questo metodo calcola la Geometric Mean tra Sensitivity e Specificity.

        Parameters
        ----------
        y_test : np.ndarray
            Etichette reali del dataset di test.
        y_pred : np.ndarray
            Etichette predette dal modello.

        Returns
        -------
        float
            Geometric Mean, valore compreso tra 0 e 1.
        """
        sensitivity = Sensitivity().calcola(y_test, y_pred)
        specificity = Specificity().calcola(y_test, y_pred)
        geometric_mean = np.sqrt(sensitivity * specificity) if sensitivity * specificity> 0 else 0
        return geometric_mean
 
class AUC(Metriche):
    """
    Questa classe calcola l'Area Under Curve (AUC) della curva ROC.
    """
    def calcola(self, y_test, y_scores, knn: KNN):
        """
        Questo metodo calcola l'AUC (Area Under the Curve) della curva ROC.
    
        Parameters
        ----------
        y_test : np.ndarray
            Etichette reali del dataset di test.
        y_scores : list
            Lista contenente i punteggi di malignità assegnati dal modello 
            a ciascun campione nel dataset di test. 
        knn : KNN
            Istanza del modello K-Nearest Neighbors.

        Returns
        -------
        float
            Valore dell'area sotto la curva ROC (AUC), compreso tra 0 e 1.
        tuple
            Contiene due liste:
            - tprs : True Positive Rates per diverse soglie.
            - fprs : False Positive Rates per diverse soglie.
        """
        tprs, fprs = knn.ROC_curve(y_test, y_scores)
        auc = np.trapz(tprs, fprs)
        return auc, (tprs, fprs)
    
            
class AllMetrics(Metriche):
    """
    Classe che calcola tutte le metriche di valutazione e la curva ROC.
    """
    def calcola(self, y_test, y_pred, y_scores, knn: KNN):
        """
        Metodo che calcola tutte le metriche e disegna la curva ROC.
        
        Parameters
        ----------
        y_test : np.ndarray
            Etichette reali del dataset di test.
        y_pred : np.ndarray
            Etichette predette dal modello.
        y_scores : list
            Lista contenente i punteggi di malignità assegnati dal modello 
            a ciascun campione nel dataset di test. 
        knn : KNN
            Istanza del modello K-Nearest Neighbors.

        Returns
        -------
        result : np.ndarray
            
        roc_data : tuple
        """
        accuracy = Accuracy().calcola(y_test, y_pred)
        error = Error().calcola(y_test, y_pred)
        sensitivity = Sensitivity().calcola(y_test, y_pred)
        specificity = Specificity().calcola(y_test, y_pred)
        geometric_mean = GeometricMean().calcola(y_test, y_pred)
        auc, roc_data = AUC().calcola(y_test, y_scores, knn=knn) 
        result = np.array([accuracy, error, sensitivity, specificity, geometric_mean, auc])
        return result, roc_data
    
