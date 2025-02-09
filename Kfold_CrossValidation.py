import numpy as np
from collections import Counter
import random

class KfoldCrossValidationKNN:

    def __init__(self, k: int, K: int):
        """ Costruttore della classe """
        self.k = k  # Numero dei vicini
        self.K = K  # Numero di fold

    def kfold_split(self, X, y):
        """ Metodo che divide e restituisce gli indici del dataset in K fold """
        indices = np.arange(len(X))
        fold_size = len(X) // self.K
        folds = []

        for i in range(self.K):
            test_indices = indices[i * fold_size: (i + 1) * fold_size] # Test indices
            train_indices = np.setdiff1d(indices, test_indices) # Train indices

            folds.append((train_indices, test_indices))
        
        return folds
    
    def euclidean_distance(self, x1: float, x2: float):
        """ Metodo che calcola la distanza euclidea tra due vettori """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def predict(self, X_train, y_train, X_test):
        y_pred = []

        for i in range(len(X_test)):
            distances = [self.euclidean_distance(X_test.values[i], x) for x in X_train.values]
            neighbour_indices = np.argsort(distances)[:self.k]
            neighbour_labels = y_train.iloc[neighbour_indices].values.flatten()

            label_counts = Counter(neighbour_labels)
            max_count = max(label_counts.values())

            # Trova tutte le classi con il massimo conteggio
            top_labels = [label for label, count in label_counts.items() if count == max_count]

            # Se c'è un pareggio, scegli casualmente tra le classi con lo stesso numero di voti
            y_pred.append(int(random.choice(top_labels)))
        return y_pred

    def evaluate(self, X: np.array, y: np.array):
        """ Metodo che valuta il modello"""
        folds = self.kfold_split(X, y)
        
        accuracy = np.zeros(self.K)
        error = np.zeros(self.K)
        sensitivity = np.zeros(self.K)
        specificity = np.zeros(self.K)
        geometric_mean = np.zeros(self.K)
        AUC = np.zeros(self.K) # da calcolare
        TP = np.zeros(self.K)
        TN = np.zeros(self.K)
        FP = np.zeros(self.K)
        FN = np.zeros(self.K)
        confusion_matrix = np.zeros((self.K, 2, 2))

        for i in range(self.K):
            train_indices, test_indices = folds[i]
            X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
            X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

            y_pred = self.predict(X_train, y_train, X_test)
            
            # Conversione per evitare problemi di confronto
            y_pred = np.array(y_pred, dtype=int)
            y_test = y_test.values.ravel().astype(int)  # Rende y_test un array 1D

            TP[i] = np.sum((y_pred == 2) & (y_test == 2))
            TN[i] = np.sum((y_pred == 4) & (y_test == 4))
            FP[i] = np.sum((y_pred == 2) & (y_test == 4))
            FN[i] = np.sum((y_pred == 4) & (y_test == 2))

            accuracy[i] = np.sum(y_pred == y_test) / len(y_test)
            error[i] = 1 - accuracy[i]  # Manteniamo il valore tra 0 e 1
            sensitivity[i] = TP[i] / np.sum(y_test == 2)
            specificity[i] = TN[i] / np.sum(y_test == 4)
            geometric_mean[i] = np.sqrt(sensitivity[i] * specificity[i])
            confusion_matrix[i] = np.array([[TN[i], FP[i]], [FN[i], TP[i]]])

        model_accuracy = np.mean(accuracy)
        model_error = np.mean(error)
        model_sensitivity = np.mean(sensitivity)
        model_specificity = np.mean(specificity)
        model_geometric_mean = np.mean(geometric_mean)
        
        return accuracy, error, sensitivity, specificity, geometric_mean, model_accuracy, model_error, model_sensitivity, model_specificity, model_geometric_mean, confusion_matrix
