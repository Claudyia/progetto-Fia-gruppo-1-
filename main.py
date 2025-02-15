from DataPreprocessing import PreprocessingDataset
from knn import KNN
from Kfold_CrossValidation import KfoldCrossValidationKNN
from HoldOut import HoldOut
from classe_random_subsampling import RandomSubsampling

if __name__ == "__main__":
    # Creiamo un'istanza della classe"
    preprocessing = PreprocessingDataset()
    # Carico dataset
    dataset_caricato = preprocessing.carica_dataset( )
    
    # Pulisco il dataset
    X, y = preprocessing.pulisci_dataset()
    
    # Normalizzo le features
    x_normalizzato = preprocessing.normalizza_features()

    if len(x_normalizzato) != len(y):
        raise ValueError("Errore: il numero di righe delle features e del target non coincide!")
    else:
        print("\nIl numero di righe delle features e del target coincide!\n")

    if len(x_normalizzato) == 0:
        raise ValueError("Errore: il numero di righe delle features è zero!")
    else:
        metodo = int(input(
        "Scegliere il metodo (digitando '1', '2' o '3') che si vuole usare per la KNN:\n"
        "1. Holdout\n"
        "2. K-fold Cross Validation\n"
        "3. Random Subsampling\n"
        ))

    if metodo == 1:
        k = int(input("Inserisci il numero di vicini (k) per l'algoritmo KNN: "))
        dim_test_set = float(input("Inserisci la dimensione del test set: "))
        holdout = HoldOut(k, dim_test_set)

        # Valutiamo il modello
        accuracy, error, sensitivity, specificity, geometric_mean, confusion_matrix = holdout.evaluate(x_normalizzato, y)

        print(f"\nAccuracy: {accuracy*100}%")
        print(f"Error: {error*100}%")
        print(f"Sensitivity: {sensitivity*100}%")
        print(f"Specificity: {specificity*100}%")
        print(f"Geometric Mean: {geometric_mean*100}%")
        print(f"Confusion Matrix:\n{confusion_matrix}")

    elif metodo == 2:
        k = int(input("Inserisci il numero di vicini (k) per l'algoritmo KNN: "))   
        K = int(input("Inserisci il numero di fold per la cross-validation: "))
        kfold = KfoldCrossValidationKNN(k, K)

        # Valutiamo il modello
        accuracy, error, sensitivity, specificity, geometric_mean, model_accuracy, model_error, model_sensitivity, model_specificity, model_geometric_mean, confusion_matrix = kfold.evaluate(x_normalizzato, y)

        print(f"\nAccuracy: {model_accuracy*100}%")
        print(f"Error: {model_error*100}%")
        print(f"Sensitivity: {model_sensitivity*100}%")
        print(f"Specificity: {model_specificity*100}%")
        print(f"Geometric Mean: {model_geometric_mean*100}%")

        for idx, matrix in enumerate(confusion_matrix):
            TN, FP = int(matrix[0, 0]), int(matrix[0, 1])
            FN, TP = int(matrix[1, 0]), int(matrix[1, 1])
            
            print(f"Confusion Matrix {idx+1}:")
            print(f"TN: {TN}, FP: {FP}")
            print(f"FN: {FN}, TP: {TP}\n")
    elif metodo == 3:
        k = int(input("Inserisci il numero di vicini (k) per l'algoritmo KNN: "))
        dim_test_set = float(input("Inserisci la dimensione del test set: "))
        K = int(input("Inserisci il numero di iterazione per il random-subsampling: "))
        
        random_subsampling = RandomSubsampling(k, K, dim_test_set)
        accuracy_media, error_rate_media, sensitivity_media, specificity_media, geometric_mean_media, confusion_matrix_iterazione = random_subsampling.evaluate(x_normalizzato, y)

        print(f"\nAccuracy: {accuracy_media*100}%")
        print(f"Error Rate: {error_rate_media*100}%")
        print(f"Sensitivity: {sensitivity_media*100}%")
        print(f"Specificity: {specificity_media*100}%")
        print(f"Geometric Mean: {geometric_mean_media*100}%")

        for idx, matrix in enumerate(confusion_matrix_iterazione):
            TN, FP = int(matrix[0, 0]), int(matrix[0, 1])
            FN, TP = int(matrix[1, 0]), int(matrix[1, 1])
            
            print(f"Confusion Matrix {idx+1}:")
            print(f"TN: {TN}, FP: {FP}")
            print(f"FN: {FN}, TP: {TP}\n")

    else: 
        raise ValueError("Errore: metodo non disponibile!")

