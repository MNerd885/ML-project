import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import lsq_linear
from joblib import Parallel, delayed # Per parallelizzare le operazioni con la CPU (lsq_linear)

# ─────────────────────────────────────────────
# 4. MODELLO RF+AR
# ─────────────────────────────────────────────

def fit_ar_per_leaf(tree: DecisionTreeRegressor,
                    X_train: np.ndarray, y_train: np.ndarray,
                    delta_y: int) -> dict:
    """
    Per ogni foglia dell'albero, fitta un modello AR lineare con vincoli
    (Problem 1 del paper) usando scipy.lsq_linear.

    Returns:
        leaf_models: dict {leaf_id: {'a': coeffs, 'f': bias}}
    """
    # Serve per raggruppare i campioni per foglia.
    leaf_ids   = tree.apply(X_train)
    leaf_models = {}

    # Per ogni foglia distinta, crea una maschera booleana per selezionare 
    # solo i campioni che appartengono a quella foglia, e li isola in X_leaf e y_leaf.
    for leaf in np.unique(leaf_ids):
        mask = (leaf_ids == leaf)
        X_leaf = X_train[mask]
        y_leaf = y_train[mask]

        # Caso degenere: se una foglia ha un solo campione, non si può fittare 
        # una regressione. Si usa la media del campione come valore costante di fallback.
        if len(y_leaf) < 2:
            # Foglia con un solo campione: usa la media come fallback
            leaf_models[leaf] = {"a": np.zeros(delta_y + 1), "f": float(y_leaf.mean())}
            continue

        # Matrice di regressione: usiamo i primi delta_y+1 lag come regressor
        n_lag_cols = min(delta_y + 1, X_leaf.shape[1])
        Lambda = X_leaf[:, :n_lag_cols]
        lam_y = y_leaf # Valore target

        # Definisce i vincoli fisici sui coefficienti AR: devono stare tra -2 e +2. 
        # Questi vincoli prevengono coefficienti troppo grandi e garantiscono stabilità 
        # del modello AR
        lb = np.full(n_lag_cols, -2.0)
        ub = np.full(n_lag_cols,  2.0)

        # Risolve il problema di regressione lineare vincolata (Problem 1 del paper) 
        # usando l'algoritmo BVLS (Bounded Variable Least Squares). result.x 
        # contiene i coefficienti ottimali trovati.
        result = lsq_linear(Lambda, lam_y, bounds=(lb, ub), method="bvls")
        coeffs = result.x

        # Salva i coefficienti AR (a, tutti tranne l'ultimo) e il bias 
        # (f, l'ultimo coefficiente) per questa foglia. 
        # Se c'è un solo coefficiente, il bias è zero.
        leaf_models[leaf] = {
            "a": coeffs[:-1] if len(coeffs) > 1 else coeffs,
            "f": coeffs[-1]  if len(coeffs) > 1 else 0.0
        }

    return leaf_models


class RFARModel:
    """
    Modello RF+AR: Random Forest con regressori AR per foglia.

    Per semplicità di bozza, l'implementazione usa scikit-learn RF
    per la struttura degli alberi e il routing, mentre i parametri AR
    per foglia sono fittati separatamente con lsq_linear.

    Per ogni stream p e ogni step j nell'orizzonte N, viene addestrato
    un modello separato: in totale p × N modelli.
    """

    def __init__(self, n_horizon: int, delta_y: int,
                 n_trees: int, min_leaf: int,
                 fit_ar_leaves: bool = True):
        self.n_horizon    = n_horizon
        self.delta_y      = delta_y
        self.n_trees      = n_trees
        self.min_leaf     = min_leaf
        self.fit_ar_leaves = fit_ar_leaves

        # Dizionario vuoto che alla fine del training conterrà la struttura 
        # models[stream][j] = (RF, leaf_models), ovvero per ogni stream e ogni 
        # step dell'orizzonte, la coppia (foresta, parametri AR per foglia).
        self.models: dict = {}

    def fit_rf_ar(self, X_train: np.ndarray, y_dict: dict) :
        """
        Addestra un RF (e i relativi modelli AR per foglia) per ogni
        combinazione (stream, step j).

        Returns:
            models: dict { 'stream'}
        """
        for stream, y_train in y_dict.items():
            self.models[stream] = {}
            for j in range(self.n_horizon):
                # Target: y(k + j + 1) — shift di j+1 passi
                # Nota: per una bozza usiamo il target diretto;
                # un'implementazione completa propagherebbe le predizioni
                y_shifted = np.roll(y_train, -(j + 1))
                y_shifted = y_shifted[:-(j + 1)]
                X_j       = X_train[:-(j + 1)]

                rf = RandomForestRegressor(
                    n_estimators=self.n_trees,
                    min_samples_leaf=self.min_leaf,
                    random_state=42,
                    oob_score=True,
                    bootstrap=True
                )
                rf.fit(X_j, y_shifted)

                leaf_models = {}
                if self.fit_ar_leaves:
                    
                    # Fit AR per ogni albero della foresta
                    # Esegue fit_ar_per_leaf su tutti gli alberi in parallelo
                    results = Parallel(n_jobs=-1)(
                        delayed(fit_ar_per_leaf)(tree_est, X_j, y_shifted, self.delta_y)
                            for tree_est in rf.estimators_
                    )

                    # Aggrega i risultati (sostituisce il vecchio ciclo for)
                    for lm in results:                          # lm = leaf_models di un albero
                        for leaf_id, params in lm.items():
                            if leaf_id not in leaf_models:
                                leaf_models[leaf_id] = {"a": [], "f": []}
                            leaf_models[leaf_id]["a"].append(params["a"])
                            leaf_models[leaf_id]["f"].append(params["f"])


                    # Media dei parametri tra alberi (Eq. 6 del paper)
                    for leaf_id in leaf_models:
                        leaf_models[leaf_id]["a"] = np.mean(leaf_models[leaf_id]["a"], axis=0)
                        leaf_models[leaf_id]["f"] = np.mean(leaf_models[leaf_id]["f"])

                self.models[stream][j] = (rf, leaf_models)

            print(f"[fit] Stream '{stream}': {self.n_horizon} modelli addestrati")

    def predict(self, X_test: np.ndarray) -> dict:
        """
        Predice i valori futuri per ogni stream e ogni step j.

        Returns:
            preds: dict {stream: np.ndarray (n_samples, n_horizon)}
        """
        preds = {}
        for stream, horizon_models in self.models.items():
            stream_preds = []
            for j in range(self.n_horizon):
                rf, leaf_models = horizon_models[j]

                if self.fit_ar_leaves and leaf_models:
                    # Usa i parametri AR per foglia
                    leaf_ids = rf.estimators_[0].apply(X_test)
                    y_pred = np.array([
                        np.dot(leaf_models.get(lid, {"a": np.zeros(1), "f": 0})["a"],
                               X_test[i, :len(
                                   leaf_models.get(lid, {"a": np.zeros(1)})["a"]
                               )]) +
                        leaf_models.get(lid, {"f": 0})["f"]
                        for i, lid in enumerate(leaf_ids)
                    ])
                else:
                    y_pred = rf.predict(X_test)

                stream_preds.append(y_pred)

            preds[stream] = np.array(stream_preds).T   # (n_samples, n_horizon)
        
        return preds
    

# ─────────────────────────────────────────────
# 5. TRAINING E PREDIZIONE  (→ vedi classe RFARModel sopra)
# ─────────────────────────────────────────────
# Il train/test split non è necessario: i due CSV sono già separati.