# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from knn import KNN

class RandomSubsampling(KNN): #eredita dalla classe KNN
    
    def __init__(self, k: int):
        """ Costruttore della classe """
        super().__init__(k)  # Chiamata al costruttore della superclasse (KNN)
        
        
    def random_subsampling(self, X: pd.DataFrame, y: pd.DataFrame, dim_test_set=0.2, K=10) -> list:
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
        if not (0 < dim_test_set < 1):
            raise ValueError("Errore: dim_test_set deve essere un valore tra 0 e 1.")
        
        # Verifico che K sia un valore valido:
        if K < 1:
            raise ValueError("Errore: K deve essere un valore intero positivo.")

        # Lista per salvare i risultati di ogni iterazione
        risultati_iterazioni = []

        for i in range(K):
            # Numero totale di campioni del dataframe X
            num_campioni = X.shape[0]

            # Calcolo il numero di campioni da destinare al test set
            num_campioni_test = int(num_campioni * dim_test_set)

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
    
    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame, K=10, dim_test_set=0.2):
        """ Metodo che valuta il modello in cui è stato utilizzato Random Subsampling """
        
        # Chiamo il metodo random_subsampling():
        risultati_iterazioni = self.random_subsampling(X, y, dim_test_set=dim_test_set, K=K) # lista di tuple (X_train, y_train, X_test, y_test)
    
        # Inizializzo le metriche:
        accuracy_totale = 0
        error_rate_totale = 0
        sensitivity_totale = 0
        specificity_totale = 0
        geometric_mean_totale = 0
        confusion_matrix_totale = np.array([[0, 0], [0, 0]])
    
        for X_train, y_train, X_test, y_test in risultati_iterazioni:
            y_pred = self.predici(X_test, X_train, y_train)  # Uso il metodo di KNN
    
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
            confusion_matrix_iterazione = np.array([[TN, FP], [FN, TP]])  # Matrice per questa iterazione
            confusion_matrix_totale += confusion_matrix_iterazione  # Sommo tutte le matrici di confusione
    
        # Calcolo la media delle metriche su tutte le iterazioni
        accuracy_media = accuracy_totale / K
        error_rate_media = error_rate_totale / K
        sensitivity_media = sensitivity_totale / K
        specificity_media = specificity_totale / K
        geometric_mean_media = geometric_mean_totale / K
    
        return accuracy_media, error_rate_media, sensitivity_media, specificity_media, geometric_mean_media, confusion_matrix_totale