# -*- coding: utf-8 -*-
import pandas as pd

def carica_dataset() -> pd.DataFrame:
    '''
    Questa funzione permette di caricare il dataset CSV, mostrando in output
    un messaggio che specifica se il caricamento del dataset è avvenuto con 
    successo o meno.
    
    Returns
    -------
    dataset: Dataframe.
    '''
    file_path = input("Inserisci il path del dataset CSV (senza apici):\n")
    
    #Gestisco l'eccezione con try ed except:
    try:
        dataset = pd.read_csv(file_path) #carico il dataset
        print("\nDataset caricato con successo!\n")
    except Exception as e:
        print(f"\nErrore nel caricamento del file: {e}\n")
        return None
    
    return dataset



def pulisci_dataset(dataset_caricato: pd.DataFrame) -> pd.DataFrame:
    """
    Questa funzione pulisce il dataset in questo modo:
    1) Se il dataset contiene numeri in formato stringa, vengono convertiti in numerico.
       Se viene trovato un valore non numerico (come per esempio uno spazio vuoto),
       lo trasforma in NaN (Not a Number).
    2) Sostituisce i valori mancanti (NaN) con la media della colonna.
    3) Rimuove eventuali righe duplicate.
    4) Stampa il numero di valori mancanti prima e dopo la pulizia per verificare eventuali errori.
    
    Parameters
    ----------
    dataset_caricato: Dataframe.
    
    Returns
    -------
    dataset_clean: Dataframe.
    """
    # Se il dataset è vuoto, interrompo la funzione
    if dataset_caricato is None:
        print("Errore: non è stato caricato nessun dataset da pulire!")
        return None
    
    #Punto 1:
    dataset_clean = dataset_caricato.apply(pd.to_numeric, errors='coerce')
    
    #Punto 4: Stampo i valori mancanti prima della pulizia
    print("\nValori mancanti prima della pulizia:")
    print(dataset_clean.isnull().sum())

    #Punto 2:
    dataset_clean.fillna(dataset_clean.mean(numeric_only=True), inplace=True)

    #Punto 3:
    dataset_clean.drop_duplicates(inplace=True)

    #Punto 4: Stampo i valori mancanti dopo la pulizia
    print("\nValori mancanti dopo la pulizia:")
    print(dataset_clean.isnull().sum())

    return dataset_clean


def separa_features_e_target(dataset_pulito: pd.DataFrame) -> tuple:
    '''
    La funzione inizia definendo una lista di colonne chiamata features. 
    Queste colonne rappresentano le features del dataset, ossia le variabili indipendenti 
    che verranno utilizzate per fare una previsione (tumore benigno o maligno).
    Successivamente viene definita la colonna che rappresenta il target 
    (la variabile dipendente che vogliamo prevedere). Il target è la colonna "classtype_v1".
    Successivamente la funzione controlla se le colonne "features" e "target" sono presenti nel dataset.
    Infine vengono separati i feutures (x) e target (y).

    Parameters
    ----------
    dataset_pulito: Dataframe.

    Returns
    -------
    Una tupla contenenti:
    - x: Dataframe
    - y: Dataframe
    '''
    # Definisco le colonne delle features
    features = [
        "clump_thickness_ty", "uniformity_cellsize_xx", "Uniformity of Cell Shape", 
        "Marginal Adhesion", "Single Epithelial Cell Size", "bareNucleix_wrong", 
        "Bland Chromatin", "Normal Nucleoli", "Mitoses"]
    
    # Definisco la colonna target
    colonna_target = ["classtype_v1"]

    # Controllo se le colonne esistono nel dataset
    for col in features + colonna_target:
        if col not in dataset_pulito.columns:
            raise ValueError(f"Colonna '{col}' non trovata nel dataset!")

    # Separo le x (features) e y (target)
    x = dataset_pulito[features]
    y = dataset_pulito[colonna_target]
    
    print("\nFeatures e target separati con successo!\n")
    
    return (x, y)


def normalizza_features(x: pd.DataFrame) -> pd.DataFrame:
    """
    Questa funzione normalizza le features utilizzando il metodo Min-Max Scaling,
    portando i valori delle features nell'intervallo [0,1].
    
    Parameters
    ----------
    x: DataFrame
        Il dataset x contiene le righe e colonne delle features.
    
    Returns
    -------
    x_normalized: DataFrame
        Il dataset con le features normalizzate.
    """
    # Calcolo Min e Max di ogni colonna
    x_min = x.min()
    x_max = x.max()

    # Applicazione della normalizzazione Min-Max
    x_normalized = (x - x_min) / (x_max - x_min)
    
    print("\nFeatures normalizzate con Min-Max Scaling!\n")
    
    return x_normalized


if __name__ == "__main__":
    #carico dataset
    dataset_caricato = carica_dataset()
    
    #pulisco il dataset
    dataset_pulito = pulisci_dataset(dataset_caricato)
    
    # Separo le features e il target
    (x, y) = separa_features_e_target(dataset_pulito)
     
    # Normalizzo le features
    x_normalizzato = normalizza_features(x)
    
    print("\n Le prime 5 righe delle features normalizzate:\n")
    print(x_normalizzato.head())

