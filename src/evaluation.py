import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# My Libraries
from rfarmodel import RFARModel
from preprocess_RFAR import *
from rfarmodel import RFARModel
from constants import N_HORIZON, DELTA_Y
# ─────────────────────────────────────────────
# 6. VALUTAZIONE
# ─────────────────────────────────────────────

def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Normalized RMSE (normalizzato per il range del segnale),
    come usato nel paper. Restituisce percentuale.
    """
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    denom = y_true.max() - y_true.min()
    if denom == 0:
        return 0.0
    return (rmse / denom) * 100


def evaluate(model: RFARModel, X_test: np.ndarray,
             y_test: dict) -> dict:
    """
    Calcola NRMSE per ogni stream e ogni step dell'orizzonte.

    Returns:
        results: dict {stream: np.ndarray (n_horizon,)}
    """
    preds   = model.predict(X_test)
    results = {}

    for stream in y_test:
        errors = []
        for j in range(model.n_horizon):
            # Allinea target e predizione (il target è shiftato di j+1)
            y_true = y_test[stream][j + 1: len(preds[stream]) + j + 1]
            y_pred = preds[stream][:len(y_true), j]
            errors.append(nrmse(y_true, y_pred))
        results[stream] = np.array(errors)
        print(f"[eval] {stream}: NRMSE N=1 → {errors[0]:.2f}% | "
              f"N=10 → {errors[-1]:.2f}%")

    return results


# ─────────────────────────────────────────────
# 7. CONFRONTO BASELINE vs ARRICCHITO
# ─────────────────────────────────────────────

def compare_and_plot(results_baseline: dict, results_enriched: dict,
                     streams: list, save_path: str = None):
    """
    Plot NRMSE per ogni stream: baseline (solo lag) vs arricchito
    (lag + feature temporali). Replica lo stile Figure 2-4 del paper.
    """
    n_streams = len(streams)
    fig, axes = plt.subplots(n_streams, 1,
                             figsize=(9, 3 * n_streams),
                             sharex=True)
    if n_streams == 1:
        axes = [axes]

    x_axis = np.arange(1, N_HORIZON + 1)

    for ax, stream in zip(axes, streams):
        if stream in results_baseline:
            ax.plot(x_axis, results_baseline[stream],
                    "o--", color="#E74C3C", linewidth=1.5,
                    label="RF+AR (solo lag — paper)")
        if stream in results_enriched:
            ax.plot(x_axis, results_enriched[stream],
                    "s-", color="#2980B9", linewidth=2,
                    label="RF+AR + feature temporali")

        ax.set_ylabel("NRMSE (%)")
        ax.set_title(stream.upper())
        ax.legend(fontsize=8)
        ax.set_xticks(x_axis)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Prediction Horizon N")
    plt.suptitle("Confronto NRMSE: Baseline vs Feature Temporali",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    plt.show()


# ─────────────────────────────────────────────
# 8. FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def plot_feature_importance(model: RFARModel, feature_names: list,
                             streams: list, top_n: int = DELTA_Y):
    """
    Visualizza le feature importance aggregate per ogni stream,
    mediando sui modelli dei diversi step dell'orizzonte.

    Utile per capire quali feature temporali contano di più
    per ogni tipo di traffico (es. is_dazn_peak per DAZN).
    """
    n_streams = len(streams)
    fig, axes = plt.subplots(1, n_streams,
                             figsize=(6 * n_streams, 5))
    if n_streams == 1:
        axes = [axes]

    for ax, stream in zip(axes, streams):
        if stream not in model.models:
            continue

        # Media delle importanze tra tutti i passi dell'orizzonte
        importances = np.zeros(len(feature_names))
        for j in range(model.n_horizon):
            rf, _ = model.models[stream][j]
            importances += rf.feature_importances_
        importances /= model.n_horizon

        # Top N feature
        idx_sorted = np.argsort(importances)[-top_n:]
        ax.barh(
            [feature_names[i] for i in idx_sorted],
            importances[idx_sorted],
            color="#2980B9", alpha=0.85
        )
        # Condizione per distinguere tra dataset di training e testing
        ax.set_title(f"Feature Importance\n{stream.upper()}")
        ax.set_xlabel("Importanza media")
        ax.grid(True, axis="x", alpha=0.3)

    plt.suptitle("Feature Importance per Stream di Traffico",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    plt.show()

def plot_fitted_model(model: RFARModel,):
    
    # Crea una griglia che si estende in base ai valori del dataset
     X_grid = np.arange(min(model.models.values()), max(model.models.values()), 0.01).reshape(-1,1)
