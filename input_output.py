from DataPreprocessing import PreprocessingDataset
import pandas as pd

def get_input_parameters():
    """Ottiene i valori di k, metodo, metrica e metriche come input dell'utente e 
    li ritorna al programma principale."""

    k = int(input("Inserire il valore di k: "))
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
    return k, metodo, metrica, metriche

    
def elabora_dataset() -> tuple:
    """Carica il dataset, lo pulisce e normalizza le features. Restituisce una tupla contenenente le features normalizzate e la colonna delle labels."""

    preprocessing = PreprocessingDataset()
    # Carica dataset
    preprocessing.carica_dataset( )
    
    # Pulisce il dataset
    X, y = preprocessing.pulisci_dataset()
    
    # Normalizza le features
    x_normalizzato = preprocessing.normalizza_features()

    if len(x_normalizzato) != len(y):
        raise ValueError("Errore: il numero di righe delle features e del target non coincide!")
    else:
        print("\nIl numero di righe delle features e del target coincide!\n")

    if len(x_normalizzato) == 0:
        raise ValueError("Errore: il numero di righe delle features è zero!")
    else:
        print("\nIl numero di righe delle features è maggiore di zero!\n")

    return x_normalizzato, y

import pandas as pd

def save_to_excel(results, metriche, metrica, metodo, filename):
    """
    Salva i dati in un file Excel chiamato come filename.
    
    :param results: i dati da salvare
    :param filename: il nome del file in cui salvare i 
    :param metriche: il dizionario delle metriche
    :param metrica: la metrica scelta da salvare
    :param metodo: il metodo di validazione scelto
    """
    results_data = {}

    if metrica == 7:
        results_data["Metriche"] = ["Accuracy %", "Error %", "Sensitivity %", "Specificity %", "Geometric Mean %", "AUC %"]
        results_data["Risultati"] = [results[0][0]*100, results[0][1]*100, results[0][2]*100, results[0][3]*100, results[0][4]*100, results[0][5]*100]
    else:
        results_data["Metrica"] = [metriche[metrica] + " %"]
        results_data["Risultati"] = [results[0]*100]

    df_results = pd.DataFrame(results_data)

    confusion_matrix_values = results[1]
    if metodo == 1:
        df_conf_matrix = pd.DataFrame(confusion_matrix_values, columns=["Neg Pred", "Pos Pred"], index=["Neg Re", "Pos Re"])
    else: 
        matrices = {}
        for i, cm in enumerate(confusion_matrix_values):
            matrices[f"Esp. {i+1}"] = [cm[0].tolist(), cm[1].tolist()]
        df_conf_matrix = pd.DataFrame(matrices)

    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        df_results.to_excel(writer, sheet_name="Metriche", index=False)
        df_conf_matrix.to_excel(writer, sheet_name="Matrici di Confusione")

    print(f"I dati sono stati salvati nel file {filename}")