# -*- coding: utf-8 -*-
import pandas as pd

class PreprocessingDataset:
    def __init__(self):
        self.dataset = None
        self.x = None
        self.y = None

    def carica_dataset(self) -> pd.DataFrame:
        '''
        Questo metodo permette di caricare il dataset CSV, mostrando in output
        un messaggio che specifica se il caricamento del dataset è avvenuto con 
        successo o meno.
        '''
        file_path = input("Inserisci il path del dataset CSV:\n")
        file_path = file_path.strip("'").strip('"')
        try:
            self.dataset = pd.read_csv(file_path)
            print("\nDataset caricato con successo!\n")
        except Exception as e:
            print(f"\nErrore nel caricamento del file: {e}\n")
        
    def pulisci_dataset(self) -> tuple:
        '''
        Questo metodo pulisce il dataset in questo modo:
        1) Converte i valori numerici in formato stringa in formato numerico.
            I valori non numerici vengono trasformati in NaN (Not a Number).
        2) Rimuove righe NaN in y e le corrispondenti in X.
        3) Rimuove eventuali righe duplicate.
        4) Separazione delle feature (X) e del target (y).
        5) Sostituisce i valori mancanti (NaN) con la media della colonna per le feature.
        6) Stampa il numero di valori mancanti prima e dopo la pulizia.
        7) Restituisce feature e target puliti.
        
        Returns
        -------
        tuple
            Una tupla contenente:
            - x: pd.DataFrame
                DataFrame delle features.
            - y: pd.DataFrame
                DataFrame del target.
        '''
        if self.dataset is None:
            print("Errore: non è stato caricato nessun dataset da pulire!")
            return None
        
        # Punto 1: Conversione a numerico (trasforma stringhe in NaN se non convertibili)
        # e stampa dei valori mancanti prima della pulizia.
        dataset_clean = self.dataset.apply(pd.to_numeric, errors='coerce')
        print("\nValori mancanti prima della pulizia:")
        print(dataset_clean.isnull().sum())
        
        # Punto 2: Separazione feature (X) e target (y)
        (self.x, self.y) = self.separa_features_e_target(dataset_clean)
        
        # Punto 3: Sostituzione dei NaN nelle feature con la media della colonna
        self.x.fillna(self.x.mean(numeric_only=True), inplace=True)
        
        # Punto 4: Rimozione delle righe duplicate
        dataset_clean = pd.concat([self.x, self.y], axis=1).drop_duplicates()
        
        # Punto 5: Rimozione delle righe con NaN in y
        dataset_clean.dropna(subset=["classtype_v1"], inplace=True)
        
        # Punto 6: Stampo i valori mancanti dopo la pulizia
        print("\nValori mancanti dopo la pulizia:")
        print(dataset_clean.isnull().sum())
        
        # Aggiorno self.x e self.y dopo la pulizia
        self.x = dataset_clean[self.x.columns]
        self.y = dataset_clean[self.y.columns]
        
        return (self.x, self.y)

    def separa_features_e_target(self, dataset_pulito: pd.DataFrame) -> tuple:
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
        dataset_pulito: pd.DataFrame
            Il dataset pulito da cui estrarre le features e il target.

        Returns
        -------
        tuple
            Una tupla contenente:
            - x: pd.DataFrame
                DataFrame delle features.
            - y: pd.DataFrame
                DataFrame del target.
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

    def normalizza_features(self) -> pd.DataFrame:
        '''
        Questa funzione normalizza le features utilizzando il metodo Min-Max Scaling,
        portando i valori delle features nell'intervallo [0,1].
        
        Returns
        -------
        x_normalized: DataFrame
            Il dataset con le features normalizzate.
        '''
        if self.x is None:
            raise ValueError("Le features non sono state caricate correttamente.")
        
        x_normalized = (self.x - self.x.min()) / (self.x.max() - self.x.min())
        print("\nFeatures normalizzate con Min-Max Scaling!\n")
        return x_normalized