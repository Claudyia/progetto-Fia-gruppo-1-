import numpy as np
from collections import Counter
import random

# Implementazione del KNN con il pattern Strategy
class KNN():

    def __init__(self, k):
        """
        Inizializza il classificatore K-Nearest Neighbors.
        
        Parametri:
        - k: numero di vicini da considerare per la classificazione.
        - distanza: oggetto che implementa la strategia di calcolo della distanza.
        """
        self.k = k
        self.distanza = None

    def distanza_euclidea(self, x1: np.array, x2: np.array) -> float:
        """
        Calcola la distanza euclidea tra due punti x1 e x2.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))


    def fit(self, X_train, y_train):
        """
        Memorizza i dati di addestramento.

        Parametri:
        - X_train: matrice delle feature dei dati di addestramento.
        - y_train: array contenente le etichette corrispondenti ai dati di addestramento.
        """
        self.X_train = X_train
        self.y_train = y_train

    def predici(self, X_test, X_train, y_train) -> list:
        """
        Effettua le predizioni sui dati di test.

        Parametri:
        - X_test: matrice delle feature dei dati di test.

        Ritorna:
        - Lista delle etichette predette per ciascun punto nel dataset di test.
        """
        self.fit(X_train, y_train)
        
        y_pred = []
        X_test_np = X_test.values
        print(X_test_np)
        X_train_np = X_train.values

        for i in range(len(X_test_np)):
            distances = [self.distanza_euclidea(X_test_np[i], x) for x in X_train_np]
            neighbour_indices = np.argsort(distances)[:self.k]
            neighbour_labels = y_train.iloc[neighbour_indices].values.flatten()

            label_counts = Counter(neighbour_labels)
            max_count = max(label_counts.values())

            # Trova tutte le classi con il massimo conteggio
            top_labels = [label for label, count in label_counts.items() if count == max_count]

            # Se c'è un pareggio, scegli casualmente tra le classi con lo stesso numero di voti
            y_pred.append(int(random.choice(top_labels)))
        return y_pred
