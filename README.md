# progetto-Fia-gruppo-1-
Lorenzo Soricone, Matteo Minotta, Claudia Lara Cordova

# Classificatore k-NN con Validazione e Metriche di Valutazione

# Descrizione

Questo progetto implementa un classificatore k-Nearest Neighbors (k-NN) con diverse strategie di validazione e metriche di valutazione. Il modello permette di classificare i dati in base ai vicini più prossimi, valutando le prestazioni con metriche standard e rappresentando i risultati tramite matrici di confusione e curve ROC.

# Requisiti

Prima di eseguire il progetto, assicurarsi di avere installati i seguenti pacchetti:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `xlsxwriter`

Possono essere tutti installati eseguendo il seguente comando sul terminale:
pip install numpy pandas matplotlib seaborn xlsxwriter

# Esecuzione

-- seguire main.py per avviare il classificatore.

-Inserire i parametri richiesti:

-Valore di k per il classificatore k-NN.

-Numero identificativo per il metodo di split del dataset desiderato: Holdout, K-Fold, o Random Subsampling.

-Dimensione percentuale del Test Set (Holdout o Random Subsampling), Numero di Fold (K-Fold) o numero di esperimenti(Random Subsampling).

-Selezione della modalità di valutazione desiderata (Accuracy, Error, Sensitivity, Specificity, Geometric Mean, AUC, Tutte le Metriche).

Il modello eseguirà la classificazione e mostrerà i risultati, inclusa la matrice di confusione e la curva ROC per ogni esperimento 
(salvare, se necessario, i grafici quando vengono mostrati).

-Alla fine del programma, sono salvati i risultati in un file Excel di cui si può specificare il nome quando richiesto.

# Test

Per verificare la correttezza delle classi e delle funzioni implementate, eseguire i test unitari con il comando:

- python -m unittest 

# Risultati
	•	Holdout: Viene generata una matrice di confusione e una curva ROC per il singolo esperimento.
	•	K-Fold: Vengono stampate tante matrici di confusione e curve ROC quanti sono i fold specificati. 
	•	Random Subsampling: Per ogni iterazione, vengono stampate la matrice di confusione e la curva ROC. 
	•	Salvataggio: I risultati finali riguardanti le metrica scelta (o tutte le metriche), incluse tutte le matrici di confusione, 
 		oltre ad essere stampati sul terminale, verranno salvati in un file Excel apposito; le curve ROC generate andrebbero, invece, salvate manualmente se necessario.

# Note

Durante l'esecuzione, chiudere ogni finestra grafica prima di procedere alla visualizzazione della successiva e infine giungere al salvataggio dei risultati finali.


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


