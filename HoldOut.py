import numpy as np
from knn import KNN

class HoldOut(KNN):

    def __init__(self, k:int, dim_test_set: float):
        """ Costruttore della classe """
        super().__init__(k)  # Chiamata al costruttore della superclasse 
        self.dim_test_set = dim_test_set  # Percentuale di dati da usare per il test set
        
    def Holdout_split(self, X, y):
            """
        Questo metodo divide il dataset in Training Set (80%) e Test Set (20%) usando il metodo Holdout.

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
            # Verifico che dim_test_set sia valido:
            if not (0 < self.dim_test_set < 1):
                raise ValueError("Errore: dim_test_set deve essere un valore tra 0 e 1.")
            
            # Numero totale di campioni (ovvero numero tot di righe)
            num_campioni = X.shape[0] 

            # Calcolo il numero di campioni da destinare al test set
            num_campioni_test = int(num_campioni * self.dim_test_set) 

            # Genero una permutazione casuale degli indici di tutti i campioni (senza seed fisso)
            indici = np.random.permutation(num_campioni) #genera una lista di indici

            # Suddivido (con lo slice) gli indici tra training set e test set
            test_indici = indici[:num_campioni_test]   # Lista indici campioni per il Test set
            train_indici = indici[num_campioni_test:]  # Restanti indici per il Training set

            x_train, x_test = X.iloc[train_indici], X.iloc[test_indici]
            y_train, y_test = y.iloc[train_indici], y.iloc[test_indici]

            return x_train, x_test, y_train, y_test
    
    def evaluate(self, X: np.array, y: np.array):
        """ Metodo che valuta il modello"""
        x_train, x_test, y_train, y_test = self.Holdout_split(X, y)
        
        y_pred = self.predici(x_test, x_train, y_train)
        
        # Conversione per evitare problemi di confronto
        y_pred = np.array(y_pred, dtype=int)
        y_test = y_test.values.ravel().astype(int) # Rende y_test un array 1D

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
