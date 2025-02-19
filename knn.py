import numpy as np
import pandas as pd
import matplotlib as plt
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


    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """
        Memorizza i dati di addestramento.

        Parametri:
        - X_train: matrice delle feature dei dati di addestramento.
        - y_train: array contenente le etichette corrispondenti ai dati di addestramento.
        """
        self.X_train = X_train
        self.y_train = y_train

    def predici(self, X_test: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.DataFrame) -> list:
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
        X_train_np = X_train.values

        for i in range(len(X_test_np)):
            distances = [self.distanza_euclidea(X_test_np[i], x) for x in X_train_np]
            neighbour_indices = np.argsort(distances)[:self.k]
            neighbour_labels = y_train.iloc[neighbour_indices].values.flatten()

            # Conta quante volte appare ciascuna classe tra i vicini più prossimi
            label_counts = Counter(neighbour_labels)
            max_count = max(label_counts.values())

            # Trova tutte le classi con il massimo conteggio
            top_labels = [label for label, count in label_counts.items() if count == max_count]

            # Se c'è un pareggio, scegli casualmente tra le classi con lo stesso numero di voti
            y_pred.append(int(random.choice(top_labels)))
        return y_pred
    
    def calcola_scores(self, X_test: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.DataFrame) -> list:
        """Metodo che calcola il punteggio di malignità per ciascun punto del dataset di test.
        Serve per il calcolo dell'eventuale curva ROC e dell'AUC associata. Il punteggio indica il grado di 'sicurezza'
        con cui il classificatore ha classificato il punto come maligno."""
        self.fit(X_train, y_train)

        y_scores = []
        X_test_np = X_test.values
        X_train_np = X_train.values

        for i in range(len(X_test_np)):
            distances = [self.distanza_euclidea(X_test_np[i], x) for x in X_train_np]
            neighbour_indices = np.argsort(distances)[:self.k]
            neighbour_labels = y_train.iloc[neighbour_indices].values.flatten()

            maligni = sum(1 for label in neighbour_labels if label == 4) 
            score = maligni/self.k #Calcolo della percentuale di vicini maligni per il singolo punto

            y_scores.append(score)
        return y_scores
    
    def ROC_curve(self, y_test, y_scores):
        """Metodo che calcola i punti della curva ROC."""
        thresholds = sorted(set(y_scores), reverse=True) # Ordina i punteggi in ordine decrescente e rimuove i duplicati
        roc_points = []

        for threshold in thresholds:
            y_pred = [4 if score >= threshold else 2 for score in y_scores] # Classifica i punti in base alla soglia

            TP = np.sum((y_pred[i] == 2) & (y_test[i] == 2) for i in range(len(y_test)))
            FN = np.sum((y_pred[i] == 4) & (y_test[i] == 2) for i in range(len(y_test)))
            TPR = TP/ (TP + FN) if (TP + FN) > 0 else 0

            TN = np.sum((y_pred[i] == 4) & (y_test[i] == 4) for i in range(len(y_test)))
            FP = np.sum((y_pred[i] == 2) & (y_test[i] == 4) for i in range(len(y_test)))
            FPR = 1 - TN / (TN + FP) if (TN + FP) > 0 else 0

            roc_points.append((FPR, TPR))

        roc_points.append((1, 1)) # Aggiunge il punto (1, 1) per chiudere la curva ROC
        roc_points.append((0, 0)) # Aggiunge il punto (0, 0) per chiudere la curva ROC
        roc_points = sorted(roc_points) # Ordina i punti in ordine crescente di FPR
        fprs, tprs = zip(*roc_points) # Separa le coordinate x e y dei punti
        return tprs, fprs
    
    def plot_ROC_Curve(self, tprs, fprs):
        """Metodo che disegna la curva ROC."""
        plt.pyplot.plot(fprs, tprs)
        plt.pyplot.plot([0, 1], [0, 1], linestyle='--', color='k')
        plt.pyplot.xlabel('False Positive Rate')
        plt.pyplot.ylabel('True Positive Rate')
        plt.pyplot.title('ROC Curve')
        plt.pyplot.show()