import pandas as pd

class PreprocessingDataset:
    def __init__(self, dataset=None):
        self.dataset = dataset
        self.x = None
        self.y = None

    def carica_dataset(self, file_path: str) -> pd.DataFrame:
        try:
            self.dataset = pd.read_csv(file_path)
            print("\nDataset caricato con successo!\n")
        except Exception as e:
            print(f"\nErrore nel caricamento del file: {e}\n")
            return None
        return self.dataset

    def pulisci_dataset(self) -> pd.DataFrame:
        if self.dataset is None:
            raise ValueError("Errore: non è stato caricato nessun dataset da pulire!")

        dataset_clean = self.dataset.apply(pd.to_numeric, errors='coerce')
        dataset_clean.drop_duplicates(inplace=True)

        # Remove rows with NaN values in the target column before separating features and target
        dataset_clean = dataset_clean.dropna(subset=["classtype_v1"])
        
        # Ensure no NaN values in the target column
        if dataset_clean["classtype_v1"].isnull().values.any():
            raise ValueError("Il target contiene valori NaN dopo la pulizia iniziale.")

        # Remove the redundant call to separa_features_e_target
        self.x, self.y = self.separa_features_e_target(dataset_clean)

        # Rimuove eventuali NaN dal target e dalle feature corrispondenti
        non_nan_mask = self.y.notna().values.ravel()
        self.x, self.y = self.x.loc[non_nan_mask], self.y.loc[non_nan_mask]

        # Sostituzione dei NaN nelle feature con la media della colonna
        self.x.fillna(self.x.mean(numeric_only=True), inplace=True)

        return pd.concat([self.x, self.y], axis=1)

    def separa_features_e_target(self, dataset_pulito: pd.DataFrame) -> tuple:
        features = [
            "clump_thickness_ty", "uniformity_cellsize_xx", "Uniformity of Cell Shape", 
            "Marginal Adhesion", "Single Epithelial Cell Size", "bareNucleix_wrong", 
            "Bland Chromatin", "Normal Nucleoli", "Mitoses"
        ]
        colonna_target = ["classtype_v1"]

        for col in features + colonna_target:
            if col not in dataset_pulito.columns:
                raise ValueError(f"Colonna '{col}' non trovata nel dataset!")

        x = dataset_pulito[features]
        y = dataset_pulito[colonna_target]

        return x, y

    def normalizza_features(self) -> pd.DataFrame:
        if self.x is None:
            raise ValueError("Le features non sono state caricate correttamente.")

        return (self.x - self.x.min()) / (self.x.max() - self.x.min())



