import numpy as np
import pandas as pd
import matplotlib as plt
from collections import Counter
import random


class KNN():
    """
    Implementazione del modello di classificazione K-Nearest Neighbors (KNN).
    Il KNN classifica un nuovo dato in base alle classi dei suoi "k" vicini più 
    prossimi nel dataset di addestramento.
    """

    def __init__(self, k):
        """
        Inizializza il classificatore K-Nearest Neighbors.
        
        Parameters
        ----------
        k : int
            Numero di vicini da considerare per la classificazione.
        """
        self.k = k
        self.distanza = None    

    def distanza_euclidea(self, x1: np.array, x2: np.array) -> float:
        """
        Calcola la distanza euclidea tra due punti x1 e x2.
        
        Returns
        -------
        float
            Distanza euclidea tra i due punti.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))


    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """
        Memorizza i dati di addestramento nel modello.

        Parameters
        ----------
        X_train : pd.DataFrame
            matrice delle feature dei dati di addestramento.
        y_train : pd.DataFrame
            array contenente le etichette corrispondenti ai dati di addestramento.
        """
        self.X_train = X_train
        self.y_train = y_train

    def predici(self, X_test: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.DataFrame) -> list:
        """
        Effettua le predizioni sui dati di test basandosi sui k vicini più prossimi.

        Parameters
        ----------
        X_test : pd.DataFrame
            Feature del dataset di test.
        X_train : pd.DataFrame
            Feature del dataset di addestramento.
        y_train : pd.DataFrame
            Etichette target corrispondenti al dataset di addestramento.

        Returns
        -------
        list
            Lista contenente le etichette predette per ciascun punto nel dataset di test.
        """
        self.fit(X_train, y_train) #Memorizza i dati di addestramento
        
        y_pred = [] #Lista per le predizioni
        X_test_np = X_test.values
        X_train_np = X_train.values

        for i in range(len(X_test_np)):
            # Calcolo la distanza tra il punto di test e tutti i punti di addestramento:
            distances = [self.distanza_euclidea(X_test_np[i], x) for x in X_train_np]
            neighbour_indices = np.argsort(distances)[:self.k] #Trova gli indici dei k vicini più prossimi
            neighbour_labels = y_train.iloc[neighbour_indices].values.flatten() #Ottiene le etichette dei vicini più prossimi

            # Conta quante volte appare ciascuna classe tra i vicini più prossimi
            label_counts = Counter(neighbour_labels)
            max_count = max(label_counts.values())

            # Trova tutte le classi con il massimo conteggio
            top_labels = [label for label, count in label_counts.items() if count == max_count]

            # Se c'è un pareggio, scegli casualmente tra le classi con lo stesso numero di voti
            y_pred.append(int(random.choice(top_labels)))
            
        return y_pred
    
    def calcola_scores(self, X_test: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.DataFrame) -> list:
        """
        Metodo che calcola il punteggio di malignità per ciascun punto del dataset di test.
        Serve per il calcolo dell'eventuale curva ROC e dell'AUC associata. Il punteggio indica il grado di 'sicurezza'
        con cui il classificatore ha classificato il punto come maligno.

        Parameters
        ----------
        X_test : pd.DataFrame
            Feature del dataset di test.
        X_train : pd.DataFrame
            Feature del dataset di addestramento.
        y_train : pd.DataFrame
            Etichette corrispondenti ai dati di addestramento.

        Returns
        -------
        list
            Lista dei punteggi di malignità per ciascun punto del dataset di test.
        """
        self.fit(X_train, y_train) #Memorizza i dati di addestramento

        y_scores = [] #Lista per i punteggi
        X_test_np = X_test.values
        X_train_np = X_train.values

        for i in range(len(X_test_np)):
            distances = [self.distanza_euclidea(X_test_np[i], x) for x in X_train_np]
            neighbour_indices = np.argsort(distances)[:self.k]
            neighbour_labels = y_train.iloc[neighbour_indices].values.flatten()
            
            # Conto il numero di vicini maligni 
            maligni = sum(1 for label in neighbour_labels if label == 4) 
            score = maligni/self.k #Calcolo della percentuale di vicini maligni per il singolo punto

            y_scores.append(score)
            
        return y_scores
    
    def ROC_curve(self, y_test, y_scores) -> tuple:
        """
        Calcola i punti della curva ROC.

        La curva ROC mostra il compromesso tra True Positive Rate (TPR) e False Positive Rate (FPR)
        per diverse soglie di decisione.

        Parameters
        ----------
        y_test : np.ndarray
            Etichette reali del dataset di test.
        y_scores : list
            Punteggi di malignità.

        Returns
        -------
        tuple
            - tprs : np.array 
               True Positive Rates (Sensibilità)
            - fprs : np.array 
               False Positive Rates
        """
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
        """
        Metodo che disegna la curva ROC.

        Parameters
        ----------
        tprs : np.array 
           True Positive Rates (Sensibilità)
        fprs : np.array 
           False Positive Rates
        """
        plt.pyplot.plot(fprs, tprs)
        plt.pyplot.plot([0, 1], [0, 1], linestyle='--', color='k')
        plt.pyplot.xlabel('False Positive Rate')
        plt.pyplot.ylabel('True Positive Rate')
        plt.pyplot.title('ROC Curve')
        plt.pyplot.show()