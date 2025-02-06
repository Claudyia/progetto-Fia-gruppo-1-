from DataPreprocessing import PreprocessingDataset

if __name__ == "__main__":
    # Creiamo un'istanza della classe
    preprocessing = PreprocessingDataset()
    
    # Carico dataset
    dataset_caricato = preprocessing.carica_dataset()
    
    # Pulisco il dataset
    dataset_pulito = preprocessing.pulisci_dataset()
    
    # Separo le features e il target
    x, y = preprocessing.separa_features_e_target(dataset_pulito)
    
    # Normalizzo le features
    x_normalizzato = preprocessing.normalizza_features()
    
    print("\nLe prime 5 righe delle features normalizzate:\n")
    print(x_normalizzato.head())