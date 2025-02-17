from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from knn import KNN

class ValidationFactory:
    """Factory per creare il metodo di validazione corretto."""
    @staticmethod
    def create_validation_method(metodo, **kwargs):
        if metodo == 1:
            return HoldOut(**kwargs)
        elif metodo == 2:
            return KfoldCrossValidationKNN(**kwargs) 
        elif metodo == 3:
            return RandomSubsampling(**kwargs)
        else:
            raise ValueError("Metodo di validazione non implementato!")
            
class ValidationStrategy(ABC):
    """Classe astratta per strategie di validazione."""

    @abstractmethod
    def split(self, X, y):
        pass

    @abstractmethod
    def evaluate(self, knn: KNN, X, y):
        pass

class HoldOut(ValidationStrategy):
    def __init__(self, dim_test_set: float):
        self.dim_test_set = dim_test_set

    def split(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Questo metodo divide il dataset in Training Set e Test Set usando il metodo Holdout.

        Parameters
        ----------
        X: DataFrame
            Features normalizzate del dataset.
        y: DataFrame
            Target 
        dim_test_set: float
            Percentuale di dati da usare per il test set (default=0.2 ovvero 20% test, 80% training).

        Returns
        -------
        x_train, x_test, y_train, y_test : tuple di DataFrame
            Le features e i rispettivi target divisi in training e test set.
        """
        # Verifico che dim_test_set sia valido:
        if not (0 < self.dim_test_set < 1):
            raise ValueError("Errore: dim_test_set deve essere un valore tra 0 e 1.")
        
        # Numero totale di campioni (ovvero numero tot di righe)
        num_campioni = X.shape[0] 

        # Calcolo il numero di campioni da destinare al test set
        num_campioni_test = int(num_campioni * self.dim_test_set)

        # Genero una permutazione casuale degli indici di tutti i campioni (senza seed fisso)
        indici = np.random.permutation(num_campioni)

        # Suddivido (con lo slice) gli indici tra training set e test set
        test_indici = indici[:num_campioni_test] # Lista indici campioni per il Test set
        train_indici = indici[num_campioni_test:] # Restanti indici per il Training set

        x_train, x_test = X.iloc[train_indici], X.iloc[test_indici]
        y_train, y_test = y.iloc[train_indici], y.iloc[test_indici]

        return x_train, x_test, y_train, y_test

    def evaluate(self, knn: KNN, X: pd.DataFrame, y: pd.DataFrame):
        """Valuta il modello KNN con il metodo Holdout."""
        x_train, x_test, y_train, y_test = self.split(X, y)

        y_pred = knn.predici(x_test, x_train, y_train)
        y_pred = np.array(y_pred, dtype=int)
        y_test = y_test.values.ravel().astype(int)

        TP = np.sum((y_pred == 2) & (y_test == 2))  
        TN = np.sum((y_pred == 4) & (y_test == 4))
        FP = np.sum((y_pred == 2) & (y_test == 4))
        FN = np.sum((y_pred == 4) & (y_test == 2))

        accuracy = np.sum(y_pred == y_test) / len(y_test)
        error = 1 - accuracy  
        sensitivity = TP / np.sum(y_test == 2)
        specificity = TN / np.sum(y_test == 4)
        geometric_mean = np.sqrt(sensitivity * specificity)
        confusion_matrix = np.array([[TN, FP], [FN, TP]])

        return accuracy, error, sensitivity, specificity, geometric_mean, confusion_matrix

class KfoldCrossValidationKNN(ValidationStrategy):
    def __init__(self, K: int):
        """ Costruttore della classe """
        self.K = K  # Numero di fold

    def split(self, X: pd.DataFrame):
        """Metodo che divide e restituisce gli indici del dataset in K fold."""
        indices = np.arange(len(X)) # Array di indici
        fold_size = len(X) // self.K # Dimensione di ogni fold
        folds = []

        for i in range(self.K):
            test_indices = indices[i * fold_size: (i + 1) * fold_size]  # Test indices
            train_indices = np.setdiff1d(indices, test_indices)  # Train indices
            folds.append((train_indices, test_indices))

        return folds

    def evaluate(self, knn: KNN, X: pd.DataFrame, y:pd.DataFrame):
        """Valuta il modello KNN con la validazione K-Fold."""
        
        # Verifico che K sia un valore valido:
        if self.K < 1:
            raise ValueError("Errore: K deve essere un valore intero positivo.")
        
        folds = self.split(X)

        accuracy = np.zeros(self.K)
        error = np.zeros(self.K)
        sensitivity = np.zeros(self.K)
        specificity = np.zeros(self.K)
        geometric_mean = np.zeros(self.K)
        confusion_matrix = np.zeros((self.K, 2, 2))

        for i in range(self.K):
            train_indices, test_indices = folds[i]
            X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
            X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

            y_pred = knn.predici(X_test, X_train, y_train) 

            # Conversione per evitare problemi di confronto
            y_pred = np.array(y_pred, dtype=int)
            y_test = y_test.values.ravel().astype(int)  

            TP = np.sum((y_pred == 2) & (y_test == 2))
            TN = np.sum((y_pred == 4) & (y_test == 4))
            FP = np.sum((y_pred == 2) & (y_test == 4))
            FN = np.sum((y_pred == 4) & (y_test == 2))

            accuracy[i] = np.sum(y_pred == y_test) / len(y_test)
            error[i] = 1 - accuracy[i]  
            sensitivity[i] = TP / np.sum(y_test == 2) if np.sum(y_test == 2) > 0 else 0
            specificity[i] = TN / np.sum(y_test == 4) if np.sum(y_test == 4) > 0 else 0
            geometric_mean[i] = np.sqrt(sensitivity[i] * specificity[i]) if sensitivity[i] * specificity[i] > 0 else 0
            confusion_matrix[i] = np.array([[TN, FP], [FN, TP]])

        model_accuracy = np.mean(accuracy)
        model_error = np.mean(error)
        model_sensitivity = np.mean(sensitivity)
        model_specificity = np.mean(specificity)
        model_geometric_mean = np.mean(geometric_mean)

        return model_accuracy, model_error, model_sensitivity, model_specificity, model_geometric_mean, confusion_matrix
    
    
class RandomSubsampling(ValidationStrategy):
    
    def __init__(self, K:int, dim_test_set: float):
        """ Costruttore della classe """
        self.K = K # Numero di iterazioni per il subsampling
        self.dim_test_set = dim_test_set # Percentuale di dati da usare per il test set
        
        
    def split(self, X: pd.DataFrame, y: pd.DataFrame) -> list:
        """
        Divide il dataset in Training Set e Test Set usando il metodo Random Subsampling.

        Parameters
        ----------
        X: DataFrame
            Features normalizzate del dataset.
        y: DataFrame
            Target (benigno o maligno).
        dim_test_set: float
            Percentuale di dati da usare per il test set.
        K: int
            Numero di iterazioni per il subsampling.

        Returns
        -------
        risultati_iterazioni: 
            Lista contenente K tuple (X_train, y_train, X_test, y_test).
        """
        # Verifico che dim_test_set sia valido:
        if not (0 < self.dim_test_set < 1):
            raise ValueError("Errore: dim_test_set deve essere un valore tra 0 e 1.")
        
        # Verifico che K sia un valore valido:
        if self.K < 1:
            raise ValueError("Errore: K deve essere un valore intero positivo.")

        # Lista per salvare i risultati di ogni iterazione
        risultati_iterazioni = []

        for i in range(self.K):
            # Numero totale di campioni del dataframe X
            num_campioni = X.shape[0]

            # Calcolo il numero di campioni da destinare al test set
            num_campioni_test = int(num_campioni * self.dim_test_set)

            # Genero una permutazione casuale degli indici delle righe del dataframe X
            indici = np.random.permutation(num_campioni)  # Lista di indici mescolati

            # Suddivido gli indici delle varie righe tra training set e test set
            test_indici = indici[:num_campioni_test]
            train_indici = indici[num_campioni_test:]

            # Divido i dati
            X_train, X_test = X.iloc[train_indici], X.iloc[test_indici]
            y_train, y_test = y.iloc[train_indici], y.iloc[test_indici]

            # Salvo i risultati di ogni iterazione
            risultati_iterazioni.append((X_train, y_train, X_test, y_test))

            print(f"Iterazione numero {i+1} completata!")

        print("\nRandom Subsampling completato!")
        return risultati_iterazioni
    
    def evaluate(self, knn: KNN, X: pd.DataFrame, y: pd.DataFrame):
        """ Metodo che valuta il modello in cui è stato utilizzato Random Subsampling """
        
        # Chiamo il metodo random_subsampling():
        risultati_iterazioni = self.split(X, y) # lista di tuple (X_train, y_train, X_test, y_test)
    
        # Inizializzo le metriche:
        accuracy_totale = 0
        error_rate_totale = 0
        sensitivity_totale = 0
        specificity_totale = 0
        geometric_mean_totale = 0
        confusion_matrix_iterazione = []

        for X_train, y_train, X_test, y_test in risultati_iterazioni:
            y_pred = knn.predici(X_test, X_train, y_train)  # Uso il metodo di KNN
    
            # Conversione 
            y_pred = np.array(y_pred, dtype=int)
            y_test = y_test.values.ravel().astype(int)
    
            # Calcolo della Matrice di Confusione:
            TP = np.sum((y_pred == 2) & (y_test == 2))  # Veri Positivi (benigni correttamente classificati)
            TN = np.sum((y_pred == 4) & (y_test == 4))  # Veri Negativi (maligni correttamente classificati)
            FP = np.sum((y_pred == 2) & (y_test == 4))  # Falsi Positivi (maligni ma in realtà benigni)
            FN = np.sum((y_pred == 4) & (y_test == 2))  # Falsi Negativi (benigni ma in realtà maligni)
    
            # Calcolo delle altre metriche:
            accuracy = np.sum(y_pred == y_test) / len(y_test)  
            error_rate = 1 - accuracy  
            sensitivity = TP / np.sum(y_test == 2) if np.sum(y_test == 2) > 0 else 0  
            specificity = TN / np.sum(y_test == 4) if np.sum(y_test == 4) > 0 else 0  
            geometric_mean = np.sqrt(sensitivity * specificity) if (sensitivity * specificity) > 0 else 0  
    
            # Aggiorno le metriche per il calcolo della media
            accuracy_totale += accuracy
            error_rate_totale += error_rate
            sensitivity_totale += sensitivity
            specificity_totale += specificity
            geometric_mean_totale += geometric_mean
    
            # Aggiorno la Matrice di Confusione Totale
            confusion_matrix_iterazione.append(np.array([[TN, FP], [FN, TP]]))  # Matrice per questa iterazione
    
        # Calcolo la media delle metriche su tutte le iterazioni
        accuracy_media = accuracy_totale / self.K
        error_rate_media = error_rate_totale / self.K
        sensitivity_media = sensitivity_totale / self.K
        specificity_media = specificity_totale / self.K
        geometric_mean_media = geometric_mean_totale / self.K
    
        return accuracy_media, error_rate_media, sensitivity_media, specificity_media, geometric_mean_media, confusion_matrix_iterazione
