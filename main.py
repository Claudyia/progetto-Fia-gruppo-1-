import pandas as pd
import numpy as np

#  Caricamento del dataset
file_path = input("inserire path dataset ")# Esempio: /Users/claudia/Desktop/Data/version_1.csv
dataset = pd.read_csv(file_path)

# Conversione a numerico (trasforma stringhe in NaN se non convertibili)
dataset = dataset.apply(pd.to_numeric, errors='coerce')

# Rimuovi righe con valori mancanti
dataset = dataset.dropna()

# Sostituire i NaN con la media:
dataset = dataset.fillna(dataset.mean(numeric_only=True))

# Controllo valori mancanti dopo la pulizia
print(dataset.isnull().sum())

# Separazione delle features (X) e della variabile target (y)
X = dataset.drop('classtype_v1', axis=1)
y = dataset['classtype_v1']

#  Normalizzazione Min-Max manuale
X_min = X.min()
X_max = X.max()

X_normalized = (X - X_min) / (X_max - X_min)

# Visualizzazione dei dati normalizzati come DataFrame (opzionale)
dataset_normalized = pd.DataFrame(X_normalized, columns=X.columns)
print(dataset_normalized.head())

#  Controllo della variabile target
print(y)


