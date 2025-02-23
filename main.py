from input_output import get_input_parameters, elabora_dataset, save_to_excel
from knn import KNN
from ValidationStrategies import ValidationFactory
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
   
    # Caricamento del dataset e normalizzazione delle feature
    x_normalizzato, y = elabora_dataset()

    # Ottengo i valori di k, metodo, metrica e metriche
    k, metodo, metrica, metriche = get_input_parameters()

    # Creazione del classificatore KNN
    knn = KNN(k)

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

    # Stampa della/e confusion matrix
    confusion_matrix_values = results[1]
    if metodo == 1: # Holdout
        print("\nMatrice di Confusione:", confusion_matrix_values)
        print("\nSalvare il grafico se necessario e chiudere la finestra per proseguire")
        plt.figure(figsize=(6,4))
        sns.heatmap(confusion_matrix_values, annot=True, fmt='d', cmap='Reds', xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
        plt.xlabel("Predetto")
        plt.ylabel("Reale")
        plt.title("Matrice di Confusione")
        plt.show()

    elif metodo in [2, 3]:  # K-Fold o Random Subsampling
        print("\nMatrici di Confusione:")
        for i, cm in enumerate(confusion_matrix_values):
            print(f"Iterazione {i + 1}:\n{cm}\n"
                  "salvare il grafico se necessario e chiudere la finestra per visualizzare la matrice di confusione successiva.")
            plt.figure(figsize=(6,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
            plt.xlabel("Predetto")
            plt.ylabel("Reale")
            plt.title("Matrice di Confusione")
            plt.show()

# Stampa della curva ROC
    roc_data = results[2]
    if metodo in [2, 3]:
        for i, roc in enumerate(roc_data):
            print(f"\nROC Curve - Esperimento {i + 1}:\n"
                  "Salvare il grafico se necessario e chiudere la finestra per preseguire.")
            knn.plot_ROC_Curve(roc[0], roc[1])
    else:
        print("\nROC Curve:\n")
        print("Salvare il grafico se necessario e chiudere la finestra per proseguire.\n")
        knn.plot_ROC_Curve(roc_data[0], roc_data[1])

    # Stampa dei risultati
    print("\nRisultati Finali:")
    if metrica == 7:
        print(f"Accuracy: {results[0][0]*100}%")
        print(f"Error: {results[0][1]*100}%")
        print(f"Sensitivity: {results[0][2]*100}%")
        print(f"Specificity: {results[0][3]*100}%")
        print(f"Geometric Mean: {results[0][4]*100}%")
        print(f"AUC: {results[0][5]*100}%")
    else:
        print(f"{metriche[metrica]}: {results[0]*100}%")

    # Salvataggio dei risultati in un file Excel
    filename = input("Inserire il nome del file Excel (senza o con estensione .xlsx) in cui salvare i risultati e la/e matrice/i di confusione: ").strip()
    if not filename.endswith(".xlsx"):
        filename += ".xlsx"
    save_to_excel(results, metriche, metrica, metodo, filename)
    print("\nFine del programma.")
