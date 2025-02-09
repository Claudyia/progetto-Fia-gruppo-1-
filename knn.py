import numpy as np

# Interfaccia base per la distanza
class Distanza():
    def calcola(self, x1, x2):
        """
        Calcola la distanza tra due punti x1 e x2.
        Questo metodo deve essere implementato nelle classi derivate.
        """
        raise NotImplementedError("Questo metodo deve essere implementato dalla classe derivata.")

# Distanza Euclidea
class DistanzaEuclidea(Distanza):
    def calcola(self, x1, x2):
        """
        Calcola la distanza euclidea tra due punti x1 e x2.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

# Implementazione del KNN con il pattern Strategy
class KNN():
    def __init__(self, k, distanza: Distanza):
        """
        Inizializza il classificatore K-Nearest Neighbors.
        
        Parametri:
        - k: numero di vicini da considerare per la classificazione.
        - distanza: oggetto che implementa la strategia di calcolo della distanza.
        """
        self.k = k
        self.distanza = distanza

    def fit(self, X_train, y_train):
        """
        Memorizza i dati di addestramento.

        Parametri:
        - X_train: matrice delle feature dei dati di addestramento.
        - y_train: array contenente le etichette corrispondenti ai dati di addestramento.
        """
        self.X_train = X_train
        self.y_train = y_train

    def predici(self, X_test):
        """
        Effettua le predizioni sui dati di test.

        Parametri:
        - X_test: matrice delle feature dei dati di test.

        Ritorna:
        - Lista delle etichette predette per ciascun punto nel dataset di test.
        """
        predizione = []
        
        for x in X_test:
            distanze = [self.distanza.calcola(x, x_train) for x_train in self.X_train]
            indici_ordinati = np.argsort(distanze)[:self.k]
            labels_più_vicine = [self.y_train[i] for i in indici_ordinati]

            labels_uniche = list(set(labels_più_vicine))
            conteggio_labels = [labels_più_vicine.count(label) for label in labels_uniche]
            predicted_label = labels_uniche[np.argmax(conteggio_labels)]
            
            predizione.append(predicted_label)
        
        return [int(p) for p in predizione]
