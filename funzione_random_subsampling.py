# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def random_subsampling(x: pd.DataFrame, y: pd.DataFrame, dim_test_set=0.2, K=10) -> list:
    """
    Questa funzione divide il dataset in Training Set e Test Set usando il metodo Random Subsampling.

    Parameters
    ----------
    x: DataFrame
        Features normalizzate del dataset.
    y: DataFrame
        Target (benigno o maligno).
    dim_test_set: float
        Percentuale di dati da usare per il test set.
    K: int
        Numero di iterazioni.

    Returns
    -------
    risultati_iterazioni: 
        Lista contenente K tuple (x_train, x_test, y_train, y_test).
    """

    # Verifico che dim_test_set sia valido
    if not (0 < dim_test_set < 1):
        raise ValueError("Errore: dim_test_set deve essere un valore tra 0 e 1.")

    # Lista per salvare i risultati di ogni iterazione
    risultati_iterazioni = []

    for i in range(K):
        # Numero totale di campioni del dataframe x
        num_campioni = x.shape[0]

        # Calcolo il numero di campioni da destinare al test set
        num_campioni_test = int(num_campioni * dim_test_set)

        # Genero una permutazione casuale degli indici delle righe del dataframe x
        indici = np.random.permutation(num_campioni) #lista di indici

        # Suddivido gli indici delle varie righe tra training set e test set
        test_indici = indici[:num_campioni_test]  
        train_indici = indici[num_campioni_test:]  

        # Divido i dati
        x_train, x_test = x.iloc[train_indici], x.iloc[test_indici]
        y_train, y_test = y.iloc[train_indici], y.iloc[test_indici]

        # Salvo i risultati di ogni singola iterazione
        risultati_iterazioni.append((x_train, y_train, x_test, y_test)) #aggiunge alla lista una tupla contenente i dataframe ottenuti da ogni singola iterazione

        print(f"Iterazione numero {i+1} completata!")

    print("\nRandom Subsampling completato!")

    return risultati_iterazioni