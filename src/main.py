"""
RF+AR Traffic Prediction Pipeline - Extended with Temporal Features
===================================================================
Basato su: "Machine Learning-based Approaches Comparison for Netflix/DAZN
Streaming and Real Traffic Prediction" (Globecom)

Estensione (Punto 5): aggiunta di feature temporali cicliche e contestuali
per migliorare la predizione, in particolare per traffico DAZN e Netflix.

Struttura:
    1. Configurazione e costanti
    2. Caricamento e preprocessing del dataset
    3. Feature engineering (lag + feature temporali)
    4. Modello RF+AR (replica del paper)
    5. Training e predizione
    6. Valutazione (NRMSE)
    7. Confronto baseline vs arricchito
    8. Feature importance
"""

# ─────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

# My Libraries
from rfarmodel import RFARModel
from preprocess_RFAR import *
from evaluation import *
from constants import N_TREES, MIN_LEAF


def main(train_path: str, test_path: str, access_point: str):
    """
    Pipeline completa con train e test su file separati,
    come forniti da Sonicatel.

    Args:
        train_path:   percorso al CSV di training
        test_path:    percorso al CSV di test
        access_point: "FA1" o "FA2" (FA2 include Netflix e DAZN)
    """
    streams = STREAMS_FA1 if access_point == "FA1" else STREAMS_FA2

    # ── 1. Carica e preprocessa ───────────────
    df_train = load_dataset(train_path, access_point)
    df_test  = load_dataset(test_path,  access_point)
    df_train = preprocess(df_train)
    df_test  = preprocess(df_test)

    # ── 2. Feature engineering ────────────────
    # Applica lag e feature temporali su train e test separatamente
    df_train = add_lag_features(df_train, streams, DELTA_Y)
    df_train_ext = add_temporal_features(df_train)
    df_test  = add_lag_features(df_test,  streams, DELTA_Y)
    df_test_ext  = add_temporal_features(df_test)

    # ── 3. Matrici feature ────────────────────
    X_tr_b, y_tr, feat_paper_lag = build_feature_matrix(
        df_train, streams, use_temporal=False)
    X_te_b, y_te, _         = build_feature_matrix(
        df_test,  streams, use_temporal=False)

    X_tr_e, _,    feat_en_tr_temp = build_feature_matrix(
        df_train_ext, streams, use_temporal=True)
    X_te_e, y_te_e,    _         = build_feature_matrix(
        df_test_ext,  streams, use_temporal=True)

    # ── 4. Training ───────────────────────────
    print("=" * 50)
    print("Training BASELINE (solo lag)")
    print("=" * 50)
    model_base = RFARModel(N_HORIZON,DELTA_Y,N_TREES,MIN_LEAF,fit_ar_leaves=True)
    model_base.fit(X_tr_b, y_tr)

    print("\n" + "=" * 50)
    print("Training ARRICCHITO (lag + feature temporali)")
    print("=" * 50)
    model_enri = RFARModel(N_HORIZON,DELTA_Y,N_TREES,MIN_LEAF,fit_ar_leaves=True)
    model_enri.fit(X_tr_e, y_tr)

    # ── 5. Valutazione ────────────────────────
    print("\n[BASELINE]")
    res_base = evaluate(model_base, X_te_b, y_te)

    print("\n[ARRICCHITO]")
    res_enrich = evaluate(model_enri, X_te_e, y_te_e)

    # ── 6. Plot confronto ─────────────────────
    compare_and_plot(res_base, res_enrich, streams)

    # ── 7. Feature importance ─────────────────
    # Importanza delle feature nel training del modello del paper
    plot_feature_importance(model_base, feat_paper_lag, streams)

    plot_feature_importance(model_enri, feat_en_tr_temp, streams)
    

    # ── 8. Riepilogo numerico ─────────────────
    
    # Per N = 1
    print("\n" + "=" * 50)
    print("RIEPILOGO NRMSE (%) @ N=1")
    print("=" * 50)
    print(f"{'Stream':<15} {'Baseline':>10} {'Arricchito':>12} {'Delta':>8}")
    print("-" * 50)
    for stream in streams:
        if stream in res_base and stream in res_enrich:
            b = res_base[stream][0]
            e = res_enrich[stream][0]
            print(f"{stream:<15} {b:>10.2f}% {e:>11.2f}% {e - b:>+7.2f}%")


    # Per N = 10
    print("\n" + "=" * 50)
    print("RIEPILOGO NRMSE (%) @ N=10")
    print("=" * 50)
    print(f"{'Stream':<15} {'Baseline':>10} {'Arricchito':>12} {'Delta':>8}")
    print("-" * 50)
    for stream in streams:
        if stream in res_base and stream in res_enrich:
            b = res_base[stream][9]
            e = res_enrich[stream][9]
            print(f"{stream:<15} {b:>10.2f}% {e:>11.2f}% {e - b:>+7.2f}%")


if __name__ == "__main__":
    TRAIN_PATH   = "datasets/SONICATEL_traffic_train.csv"  # <- modifica con il tuo percorso
    TEST_PATH    = "datasets/SONICATEL_traffic_test.csv"   # <- modifica con il tuo percorso
    #ACCESS_POINT = "FA1"                           
    ACCESS_POINT = "FA2"                         

    main(TRAIN_PATH, TEST_PATH, ACCESS_POINT)
