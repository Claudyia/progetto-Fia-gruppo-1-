from DataPreprocessing import PreprocessingDataset
from knn import KNN
from ValidationStrategies import ValidationFactory
import matplotlib.pyplot as plt
import seaborn as sns

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

    metrica = int(input(
            "Scegliere la metrica (digitando '1', '2', '3', '4', '5', '6' o '7') che si vuole calcolare per valutare il modello:\n"
            "1. Accuracy\n"
            "2. Error\n"
            "3. Sensitivity\n"
            "4. Specificity\n"
            "5. Geometric Mean\n"
            "6. AUC\n"
            "7. Tutte le metriche\n"
            ))
   
    metriche = {
        1: "Accuracy",
        2: "Error",
        3: "Sensitivity",
        4: "Specificity",
        5: "Geometric Mean",
        6: "AUC",
        7: "AllMetrics"
    }

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
    results = validation_method.evaluate(knn, x_normalizzato, y, metrica)

    # Stampa dei risultati
    print("\nRisultati:")
    if metrica == 7:
        print(f"Accuracy: {results[0][0]*100}%")
        print(f"Error: {results[0][1]*100}%")
        print(f"Sensitivity: {results[0][2]*100}%")
        print(f"Specificity: {results[0][3]*100}%")
        print(f"Geometric Mean: {results[0][4]*100}%")
        print(f"AUC: {results[0][5]*100}%")
        confusion_matrix_values = results[1]
    else:
        print(f"{metriche[metrica]}: {results[0]*100}%")
        confusion_matrix_values = results[1]

    # Stampa della/e confusion matrix
    if metodo == 1: # Holdout
        print("\nConfusion Matrix - Holdout:\n", confusion_matrix_values)
        plt.figure(figsize=(6,4))
        sns.heatmap(confusion_matrix_values, annot=True, fmt='d', cmap='Reds', xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
        plt.xlabel("Predetto")
        plt.ylabel("Reale")
        plt.title("Matrice di Confusione")
        plt.show()

    elif metodo in [2, 3]:  # K-Fold o Random Subsampling
        print("\nConfusion Matrices:")
        for i, cm in enumerate(confusion_matrix_values):
            print(f"Iteration {i + 1}:\n{cm}")
            plt.figure(figsize=(6,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
            plt.xlabel("Predetto")
            plt.ylabel("Reale")
            plt.title("Matrice di Confusione")
        plt.show()