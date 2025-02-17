from DataPreprocessing import PreprocessingDataset
from knn import KNN
from ValidationStrategies import ValidationFactory 

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
        print("\nIl numero di righe delle features è maggiore di zero!\n")

    k = int(input("Inserire il valore di k: "))
    knn = KNN(k)

    metodo = int(input(
            "Scegliere il metodo (digitando '1', '2' o '3') che si vuole usare per la KNN:\n"
            "1. Holdout\n"
            "2. K-fold Cross Validation\n"
            "3. Random Subsampling\n"
    ))

    # Creazione del metodo di validazione
    if metodo == 1:
        dim_test_set = float(input("Inserire la dimensione del test set (in percentuale): "))
        validation_method = ValidationFactory.create_validation_method(metodo, dim_test_set = dim_test_set)
    elif metodo == 2:
        K = int(input("Inserire il numero di fold: "))
        validation_method = ValidationFactory.create_validation_method(metodo, K=K)
    elif metodo == 3:
        dim_test_set = float(input("Inserire la dimensione del test set (in percentuale): "))
        K = int(input("Inserire il numero di iterazioni per il random subsampling: "))
        validation_method = ValidationFactory.create_validation_method(metodo, dim_test_set = dim_test_set, K = K)
    else:
        raise ValueError("Metodo non valido!")

    # Valutazione del modello
    results = validation_method.evaluate(knn, x_normalizzato, y)

    print("\nRisultati della valutazione del modello:")
    print("Accuracy: ", results[0]*100, "%")
    print("Error: ", results[1]*100, "%")
    print("Sensitivity: ", results[2]*100, "%")
    print("Specificity: ", results[3]*100, "%")
    print("Geometric Mean: ", results[4]*100, "%")
    
    # Gestione della confusion matrix in base al metodo di validazione
    if metodo == 1:
        print("\nConfusion Matrix - Holdout:\n", results[5])

    elif metodo in [2, 3]:  # K-Fold o Random Subsampling
        print("\nConfusion Matrices:")
        for i, cm in enumerate(results[5]):
            print(f"Iteration {i + 1}:\n{cm}")

