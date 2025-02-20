# -*- coding: utf-8 -*-
import pandas as pd

class PreprocessingDataset:
    """
    Classe per la gestione del preprocessing dei dati. 
    Contiene metodi per:
    1) Caricare un dataset CSV.
    2) Pulire i dati (gestione valori mancanti, conversioni, rimozione duplicati).
    3) Separare features (x) e target (y).
    4) Normalizzare le features per migliorare la performance del modello.
    """
    def __init__(self):
        """
        Inizializzo la classe con dataset, features (x) e target (y) impostati a None.
        """
        self.dataset = None
        self.x = None
        self.y = None

    def carica_dataset(self) -> pd.DataFrame:
        '''
        Questo metodo permette di caricare il dataset CSV, mostrando in output
        un messaggio che specifica se il caricamento del dataset è avvenuto con 
        successo o meno.
        '''
        # Chiedo all'utente di inserire il percorso del dataset CSV
        file_path = input("Inserisci il path del dataset CSV:\n")
        file_path = file_path.strip("'").strip('"') #Rimuove eventuali apici del path
        
        try:
            self.dataset = pd.read_csv(file_path) #Carico il dataset
            print("\nDataset caricato con successo!\n")
        except Exception as e:
            print(f"\nErrore nel caricamento del file: {e}\n") #gestisco eventuale eccezione
        
    def pulisci_dataset(self) -> tuple:
        '''
        Questo metodo pulisce il dataset in questo modo:
        1) Converte i valori stringa in numerici.
           I non convertibili (es. spazi vuoti) diventano NaN, ovvero not a number.
        2) Separazione delle feature (x) e del target (y).
        3) Sostituisce i valori (NaN) nelle features con la media della colonna.
        4) Rimuove eventuali righe duplicate.
        5) Rimuove le righe con valori NaN nella colonna target (`classtype_v1`) e le corrispondenti in x.
        6) Stampa il numero di valori mancanti prima e dopo la pulizia.
        7) Restituisce feature e target puliti.
        
        Returns
        -------
        tuple
            Una tupla contenente:
            - x: pd.DataFrame
                DataFrame delle features pulite.
            - y: pd.DataFrame
                DataFrame del target pulito.
        '''
        if self.dataset is None:
            print("Errore: non è stato caricato nessun dataset da pulire!")
            return None
        
        # Punto 1: Conversione a numerico (trasforma stringhe in NaN se non convertibili)
        dataset_clean = self.dataset.apply(pd.to_numeric, errors='coerce')
        
        # Stampo il numero di valori NaN prima della pulizia
        print("\nValori mancanti prima della pulizia:")
        print(dataset_clean.isnull().sum())
        
        # Punto 2: Separazione feature (x) e target (y)
        (self.x, self.y) = self.separa_features_e_target(dataset_clean)
        
        # Punto 3: Sostituzione dei NaN nelle features con la media della colonna
        self.x.fillna(self.x.mean(numeric_only=True), inplace=True)
        
        # Punto 4: Rimozione delle righe duplicate
        dataset_clean = pd.concat([self.x, self.y], axis=1).drop_duplicates()
        
        # Punto 5: Rimozione delle righe con NaN nel target y
        dataset_clean.dropna(subset=["classtype_v1"], inplace=True)
        
        # Punto 6: Stampo i valori mancanti dopo la pulizia
        print("\nValori mancanti dopo la pulizia:")
        print(dataset_clean.isnull().sum())
        
        # Aggiorno self.x e self.y dopo la pulizia:
        self.x = dataset_clean[self.x.columns]
        self.y = dataset_clean[self.y.columns]
        
        return (self.x, self.y)

    def separa_features_e_target(self, dataset_pulito: pd.DataFrame) -> tuple:
        '''
        Questo metodo separa le features (x) e il target (y) dal dataset in questo modo:
            1) Definisce le colonne delle features (variabili indipendenti);
            2) Definisce la colonna target `classtype_v1` (variabile dipendente);
            3) Controlla che le colonne delle "features" e "target" esistano nel dataset;
            4) Separa features (x) e target (y);

        Parameters
        ----------
        dataset_pulito: pd.DataFrame
            Il dataset da cui estrarre le features e il target.

        Returns
        -------
        tuple
            Una tupla contenente:
            - x: pd.DataFrame
                DataFrame delle features.
            - y: pd.DataFrame
                DataFrame del target.
        '''
        # Definisco le colonne delle features:
        features = [
            "clump_thickness_ty", "uniformity_cellsize_xx", "Uniformity of Cell Shape", 
            "Marginal Adhesion", "Single Epithelial Cell Size", "bareNucleix_wrong", 
            "Bland Chromatin", "Normal Nucleoli", "Mitoses"]
        
        # Definisco la colonna target:
        colonna_target = ["classtype_v1"]

        # Controllo se le colonne esistono nel dataset:
        for col in features + colonna_target:
            if col not in dataset_pulito.columns:
                raise ValueError(f"Colonna '{col}' non trovata nel dataset!")

        # Separo le x (features) e y (target):
        x = dataset_pulito[features]
        y = dataset_pulito[colonna_target]
        
        print("\nFeatures e target separati con successo!\n")
    
        return (x, y)

    def normalizza_features(self) -> pd.DataFrame:
        '''
        Questo metodo normalizza le features utilizzando il metodo Min-Max Scaling,
        portando i valori delle features nell'intervallo [0,1].
        
        Returns
        -------
        x_normalized: DataFrame
            Il dataset con le features normalizzate.
        '''
        if self.x is None:
            raise ValueError("Le features non sono state caricate correttamente.")
        
        # Normalizzazione Min-Max: porta i valori tra 0 e 1
        x_normalized = (self.x - self.x.min()) / (self.x.max() - self.x.min())
        
        print("\nFeatures normalizzate con Min-Max Scaling!\n")
        return x_normalized