## Ragionamento sul da farsi

In questo progetto devo mostrare come le previsioni fatte sul traffico migliorino usando l'approccio con RT & RF sfruttando il modello autoregressivo che devo costruire a partire dal dataset per ogni Target e soprattutto voglio mostrare come la scelta di quanti campioni devo avere per foglia influisca sull'OOB error e congiuntamente voglio mostrare come il NRMSE diminuisca all'aumentare degli alberi. Innanzitutto devo definire quali sono le feature del dataset che utilizzo per predire il traffico di rete dei Fiber Access :

1. Le **Feature** sono le caratteristiche che descrivono il sistema, in questo caso sono: *year*, *month*, *day*, *hour*, *min*, *dayweek*, *IN*, *OUT* .
2. Le variabili **Target** sono le tipologie di traffico che si vuole prevedere, oovvero: *VOIP*, *Netflix*, *Dazn* .

Ora un altro aspetto importante da definire sono i parametri essenziali per tutta la trattazione:
- **LAG:** Rappresenta il numero di ritardi da applicare ai target nella costruzione del modello autoregressivo per ciascuno di essi. Sceglierò un lag massimo e progressivamente per fare la mia analisi aumenterò il numero di lag fino al massimo considerando un numero di colonne maggiore del mio dataset con le sole feature.
- **N_TREES:** Rappresenta il numero di alberi da far crescere nella Random Forest, utile per testare come descresce il RMSE al crescere del numero di alberi.
- **MIN_LEAF:** E' il numero minimo di campioni per ogni foglia dell'albero.

#### Creazione dei modelli autoregressivi per ogni Target

Ogni target avrà la sua matrice lag che descrive proprio i dati passati per un certo numero di lag associato a ciascun target, ciò è molto utile perché si possono sfruttare informazioni passate del sistema per predire la sua evoluzione futura.
#### Pulizia del dataset arriccchito con i lag

L'obiettivo è di eliminare i valori NaN o sostiruirli (introdotti dai lag inseriti), ciò accade proprio perché ritardando i valori del target nel tempo non si hanno informazioni precedenti riguardo l'evoluzione del modello.

Per la sostituzione di tali valori provo ad interpolare i valori NaN.
Per farlo siccome si può lavorare solamente su vettori devo sicuramente creare un ciclo for dentro il quale per ogni colonna vado a vedere quali sono i valori NaN e interpolo i valori NaN di quella colonna.


#### Separazione delle feature dai target

I lag aggiunti sono nuove feature!! Devo farci attenzione, giustamente sono dei dati del passato derivanti dal modello autoregressivo che sto usando, perciò fanno parte delle altre feature già definite. Ora per fare tale separazione devo scegliere le feature con estrazioni uniformemente distribuite e non considerando i 3 target presenti nel dataset (datasample). E prima di questo soprattuto devo far in modo di costruire una matrice di regressori ad ogni step per mostrare tutti i lag fino al massimo e fare training e testing per ogni quantità di lag che si vuile introdurre.

Per fare il train sul dataset ottenuto di sole feature scelte
#### Valutazione delle performance e test dei RTs

Per valutare performance del mio predittore vedo come questo predice i 3 target con un dataset di test e per ciascuno di essi valuto il NRMSE di ciascun RT, scelgo quello normalizzato in modo che se ho unità di misura diverse nel mio dataset 

#### Approccio con Random Forests

Innanzitutto posso far vedere come il numero di alberi che si sceglie per la RF influisce nell'ottenimento di un NRMSE in fase di testing che è progressivamente più piccolo, considerando l'influenza del numero di lag.