from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from knn import KNN
from calcolo_metriche import MetricheFactory

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

    def compute_confusion_matrix(self, y_test, y_pred) -> np.ndarray:
        """Calcola la Matrice di Confusione."""
        TP = np.sum((y_pred == 2) & (y_test == 2))
        TN = np.sum((y_pred == 4) & (y_test == 4))
        FP = np.sum((y_pred == 2) & (y_test == 4))
        FN = np.sum((y_pred == 4) & (y_test == 2))
        return np.array([[TN, FP], [FN, TP]])

class HoldOut(ValidationStrategy):
    def __init__(self, dim_test_set: float):
        self.dim_test_set = dim_test_set

    def split(self, X: pd.DataFrame, y: pd.DataFrame)-> tuple:
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

    def evaluate(self, knn: KNN, X: pd.DataFrame, y: pd.DataFrame, metrica: int) -> tuple:
        """Valuta il modello KNN con il metodo Holdout."""
        x_train, x_test, y_train, y_test = self.split(X, y)

        y_pred = knn.predici(x_test, x_train, y_train) # Predizione
        y_pred = np.array(y_pred, dtype=int) # Conversione per evitare problemi di confronto
        y_scores = knn.calcola_scores(x_test, x_train, y_train) # Calcolo dei punteggi di malignità
        y_test = y_test.values.ravel().astype(int)

        confusion_matrix = self.compute_confusion_matrix(y_test, y_pred)
        if metrica == 6:
            result, roc_data = MetricheFactory.create_metriche(metrica).calcola(y_test, y_scores, knn = knn)
            return result, confusion_matrix, roc_data
        elif metrica == 7:
            result, roc_data = MetricheFactory.create_metriche(metrica).calcola(y_test, y_pred, y_scores=y_scores, knn=knn)
            return result, confusion_matrix, roc_data
        else:
            result = MetricheFactory.create_metriche(metrica).calcola(y_test, y_pred)
        return result, confusion_matrix
    
class KfoldCrossValidationKNN(ValidationStrategy):
    def __init__(self, K: int):
        """ Costruttore della classe """
        self.K = K  # Numero di fold

    def split(self, X: pd.DataFrame, y: pd.DataFrame) -> list:
        """Metodo che divide e restituisce gli indici del dataset in K fold."""
        indices = np.arange(len(X)) # Array di indici
        fold_size = len(X) // self.K # Dimensione di ogni fold
        folds = []

        for i in range(self.K):
            test_indices = indices[i * fold_size: (i + 1) * fold_size]  # Test indices
            train_indices = np.setdiff1d(indices, test_indices)  # Train indices
            folds.append((train_indices, test_indices))

        return folds

    def evaluate(self, knn: KNN, X: pd.DataFrame, y:pd.DataFrame, metrica: int)-> tuple:
        """Valuta il modello KNN con la validazione K-Fold."""
        
        # Verifico che K sia un valore valido:
        if self.K < 1:
            raise ValueError("Errore: K deve essere un valore intero positivo.")
        
        folds = self.split(X, y)

        results = []
        confusion_matrices = []
        roc_datas = []

        for i in range(self.K):
            train_indices, test_indices = folds[i]
            X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
            X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

            y_pred = knn.predici(X_test, X_train, y_train) 
            y_scores = knn.calcola_scores(X_test, X_train, y_train) # Calcolo dei punteggi di malignità

            # Conversione per evitare problemi di confronto
            y_pred = np.array(y_pred, dtype=int)
            y_test = y_test.values.ravel().astype(int)  

            # Calcolo della Matrice di Confusione:
            confusion_matrices.append(self.compute_confusion_matrix(y_test, y_pred))

            # Calcolo delle metriche richieste
            if metrica == 6:
                result, roc_data = MetricheFactory.create_metriche(metrica).calcola(y_test, y_scores, knn = knn)
                results.append(result)
                roc_datas.append(roc_data)
            elif metrica == 7:
                result, roc_data = MetricheFactory.create_metriche(metrica).calcola(y_test, y_pred, y_scores=y_scores, knn = knn)
                results.append(result)
                roc_datas.append(roc_data)
            else:
                results.append(MetricheFactory.create_metriche(metrica).calcola(y_test, y_pred))
        
        # Calcolo della media delle metriche nel caso in cui vengano richieste tutte le metriche e non
        if metrica == 7:
            results_mean = []
            for i in range(len(results[0])):
                results_mean.append(np.mean([results[j][i] for j in range(self.K)]))
            return results_mean, confusion_matrices, roc_datas
        elif metrica == 6:
            return np.mean(results), confusion_matrices, roc_datas
        else:
            return np.mean(results), confusion_matrices
    
    
class RandomSubsampling(ValidationStrategy):

    def __init__(self, K:int, dim_test_set: float):
        """ Costruttore della classe """
        # Verifico che K sia un valore valido:
        if K < 1:
            raise ValueError("Errore: K deve essere un valore intero positivo.")
        
        # Verifico che dim_test_set sia valido:
        if not (0 < dim_test_set < 1):
            raise ValueError("Errore: dim_test_set deve essere un valore tra 0 e 1.")
        
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
    
    def evaluate(self, knn: KNN, X: pd.DataFrame, y: pd.DataFrame, metrica: int)-> tuple:
        """ Metodo che valuta il modello in cui è stato utilizzato Random Subsampling """
        
        # Chiamo il metodo random_subsampling():
        risultati_iterazioni = self.split(X, y) # lista di tuple (X_train, y_train, X_test, y_test)
    
        # Inizializzo le metriche:
        results = []
        confusion_matrices = []
        roc_datas = []

        for X_train, y_train, X_test, y_test in risultati_iterazioni:
            y_pred = knn.predici(X_test, X_train, y_train)  # Uso il metodo di KNN
    
            # Conversione 
            y_pred = np.array(y_pred, dtype=int)
            y_test = y_test.values.ravel().astype(int)
            y_scores = knn.calcola_scores(X_test, X_train, y_train) # Calcolo dei punteggi di malignità
    
            # Calcolo della Matrice di Confusione:
            confusion_matrices.append(self.compute_confusion_matrix(y_test, y_pred))

           # Calcolo delle metriche richieste 
            if metrica == 6:
                result, roc_data = MetricheFactory.create_metriche(metrica).calcola(y_test, y_scores, knn = knn)
                results.append(result)
                roc_datas.append(roc_data)
            elif metrica == 7:
                result, roc_data = MetricheFactory.create_metriche(metrica).calcola(y_test, y_pred, y_scores=y_scores, knn=knn)
                results.append(result)
                roc_datas.append(roc_data)
            else:
                results.append(MetricheFactory.create_metriche(metrica).calcola(y_test, y_pred))

        # Calcolo della media delle metriche nel caso in cui vengano richieste tutte le metriche e non
        if metrica == 7:
            results_mean = []
            for i in range(len(results[0])):
                results_mean.append(np.mean([results[j][i] for j in range(self.K)]))
            return results_mean, confusion_matrices, roc_datas
        elif metrica == 6:
            return np.mean(results), confusion_matrices, roc_datas
        else:
            return np.mean(results), confusion_matrices