# progetto-Fia-gruppo-1-
Lorenzo Soricone, Matteo Minotta, Claudia Lara Cordova

# Classificatore k-NN con Validazione e Metriche di Valutazione

# Descrizione

Questo progetto implementa un classificatore k-Nearest Neighbors (k-NN) con diverse strategie di validazione e metriche di valutazione. Il modello permette di classificare i dati in base ai vicini più prossimi, valutando le prestazioni con metriche standard e rappresentando i risultati tramite matrici di confusione e curve ROC.

# Requisiti

Prima di eseguire il progetto, assicurarsi di avere installati i seguenti pacchetti:

- pip install numpy pandas matplotlib seaborn openpyxl xlsxwriter collections

# Esecuzione

-- seguire main.py per avviare il classificatore.

-Inserire i parametri richiesti:

-Valore di k per il classificatore k-NN.

-Metodo di validazione: Holdout, K-Fold, o Random Subsampling.

-Numero di Fold (K-Fold) o interazioni(Random Subsampling).


-Scelta della metrica di valutazione (Accuracy, Sensitivity, AUC, ecc.).

Il modello eseguirà la classificazione e mostrerà i risultati, inclusa la matrice di confusione e la curva ROC.

-Salvare i risultati in un file Excel specificando il nome del file quando richiesto.

# Test

Per verificare la correttezza delle metriche implementate, eseguire i test unitari con il comando:

- python -m unittest 

# Risultati
	•	Holdout: Viene generata una matrice di confusione e una curva ROC per l’iterazione di validazione, in base al numero k di vicini scelto dall’utente.
	•	K-Fold: Vengono stampate tante matrici di confusione e curve ROC quanti sono i fold specificati. L’utente dovrà fornire sia il valore di k (numero di vicini) che il numero di fold da utilizzare.
	•	Random Subsampling: Per ogni iterazione, vengono stampate la matrice di confusione e la curva ROC. L’utente fornirà il valore di k e il numero di iterazioni desiderato.
	•	Salvataggio: I risultati finali, incluse tutte le matrici di confusione e le curve ROC, verranno salvati in un file Excel per un’ulteriore analisi.

# Note

Durante l'esecuzione, chiudere ogni finestra grafica prima di procedere alla visualizzazione della successiva.


# ESEMPI  DI OUTPUT

HOLDOUT
k=3
dimensione del test set= 0,2


![HOLDOUT](https://github.com/user-attachments/assets/3aa1cd71-7444-44d8-9fd2-80f313929f41)

![ROC HOLDOUT](https://github.com/user-attachments/assets/d6845333-e687-4b9b-8622-f66ae3acc989)

K-FOLD CROSS VALIDATION
k=3
numero di fold=2
![K FOLD1](https://github.com/user-attachments/assets/2ce3ae7d-1fb0-426e-8c29-765358f16c5d)
![K FOLD 2](https://github.com/user-attachments/assets/c5f9b665-5ff5-436c-8dd4-777bbcbe8a03)
![ROC1 KFOLD](https://github.com/user-attachments/assets/579a8017-c7b7-4715-9cb8-28591cc3ad74)
![ROC2 KFOLD](https://github.com/user-attachments/assets/9c44b53b-da8c-4c70-9ee2-445df5069ac1)

RANDOM SUBSAMPLING
k=3
dimensione del test set= 0,2
numero di iterazioni = 2
![RANDOM](https://github.com/user-attachments/assets/2782323d-d558-408d-9d1b-92c6bb60c2b0)

![RANDOM2](https://github.com/user-attachments/assets/e07f4908-2e6b-4304-9f85-560bda56c1d3)

![ROC1RANDOM](https://github.com/user-attachments/assets/fa5bb460-9ae4-41bd-a7d0-2e26657f7a43)

![ROC2 RANDOM](https://github.com/user-attachments/assets/a6aca6cd-8347-4b89-b15d-b9b3b1dd9142)


