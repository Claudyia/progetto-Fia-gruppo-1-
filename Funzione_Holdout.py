# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def holdout_method(x: pd.DataFrame, y: pd.DataFrame, dim_test_set = 0.2) -> tuple:
    """
    Questa funzione divide il dataset in Training Set (80%) e Test Set (20%) usando il metodo Holdout.

    Parameters
    ----------
    x: DataFrame
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
    # Verifico che dim_test_set sia valido
    if not (0 < dim_test_set < 1):
        raise ValueError("Errore: dim_test_set deve essere un valore tra 0 e 1.")


    # Numero totale di campioni (ovvero numero tot di righe)
    num_campioni = x.shape[0]

    # Calcolo il numero di campioni da destinare al test set
    num_campioni_test = int(num_campioni * dim_test_set)

    # Genero una permutazione casuale degli indici di tutti i campioni (senza seed fisso)
    indici = np.random.permutation(num_campioni) #genera una lista di indici

    # Suddivido (con lo slice) gli indici tra training set e test set
    test_indici = indici[:num_campioni_test]   # Lista indici campioni per il Test set
    train_indici = indici[num_campioni_test:]  # Restanti indici per il Training set

    # Divido i dati:
    '''
    .iloc[] è un metodo di Pandas per selezionare righe e colonne usando gli 
    indici numerici.
    '''
    x_train, x_test = x.iloc[train_indici], x.iloc[test_indici]
    y_train, y_test = y.iloc[train_indici], y.iloc[test_indici]

    print("\nHoldout Method completato!")
    print(f"Training set: {x_train.shape[0]} campioni")
    print(f"Test set: {x_test.shape[0]} campioni")

    return x_train, x_test, y_train, y_test
