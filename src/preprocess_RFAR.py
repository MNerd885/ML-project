import numpy as np
import pandas as pd
import holidays

from constants import STREAMS_FA1, STREAMS_FA2, SAMPLING, DELTA_Y

# ─────────────────────────────────────────────
# 2. CARICAMENTO E PREPROCESSING
# ─────────────────────────────────────────────

def load_dataset(filepath: str, access_point: str) -> pd.DataFrame:
    """
    Carica il dataset Sonicatel da CSV.

    Formato reale del CSV (colonne):
        year, month, day, hour, min, dayweek, IN, OUT, VOIP, Netflix, DAZN

    - La data è spezzata in colonne separate → viene ricostruita come
      DatetimeIndex a frequenza 5min.
    - dayweek: 1=lunedì ... 7=domenica (non usata direttamente, ricavata
      dall'indice datetime per coerenza con pandas).

    Args:
        filepath:     percorso al file CSV (train o test)
        access_point: "FA1" (IN, OUT, VOIP) o "FA2" (+ Netflix, DAZN)

    Returns:
        DataFrame con indice DatetimeIndex a frequenza 5min
    """
    df = pd.read_csv(filepath)

    # Crea la colonna timestamp da colonne separate che vengono estratte in linea 35 per poi rinominare la colonna min
    df["timestamp"] = pd.to_datetime(
        df[["year", "month", "day", "hour", "min"]].rename(
            columns={"min": "minute"}
        )
    )
    df = df.set_index("timestamp").sort_index() # Il dataset viene riordinato

    # Rimuove colonne di data e dayweek (non più necessarie perché sono racchiuse in timestamp)
    df = df.drop(columns=["year", "month", "day", "hour", "min", "dayweek"],
                 errors="ignore")

    # Seleziona gli stream rilevanti
    streams = STREAMS_FA1 if access_point == "FA1" else STREAMS_FA2
    
    # Mantiene o cancella le colonne relative dal tipo di strems scelto (FA1 o FA2)
    df = df[[c for c in streams if c in df.columns]]

    # Forza la frequenza a 5 minuti (segnala eventuali buchi)
    df = df.asfreq(SAMPLING)

    print(f"[load] {access_point} — {filepath.split('/')[-1]}: "
          f"{len(df)} campioni | "
          f"{df.index.min().strftime('%Y-%m-%d')} → "
          f"{df.index.max().strftime('%Y-%m-%d')}")
    print(f"[load] Missing values: {df.isna().sum().sum()} totali\n")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pulizia base:
      - Interpolazione lineare per eventuali missing values
        (max 3 campioni contigui = 15 minuti)
      - Clip dei valori negativi (artefatti di raccolta)

    Il dataset Sonicatel non ha missing values, quindi questa funzione
    è principalmente un safety net.
    """
    # Conta tutti i valori NaN nel DataFrame: il primo .sum()
    # somma per colonna, il secondo somma tutti i totali per colonna 
    # ottenendo un singolo numero.
    n_missing = df.isna().sum().sum()

    if n_missing > 0:
        df = df.interpolate(method="linear", limit=3)
        df = df.dropna()
        print(f"[preprocess] Interpolati/rimossi {n_missing} missing values")

    # Porta a zero tutti i valori negativi. Nel traffico di rete i valori negativi
    # sono artefatti di raccolta dati e non hanno senso fisico.
    df = df.clip(lower=0)

    print(f"[preprocess] {len(df)} campioni pronti\n")
    return df


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame, streams: list, delta_y: int = DELTA_Y) -> pd.DataFrame:
    """
    Crea il vettore di stato x(k) = [y(k), y(k-1), ..., y(k-delta_y)]
    per ogni stream, come nel paper.
    """

    # Per ogni stream e per ogni valore di lag da 1 a DELTA_Y, crea una nuova colonna
    # che contiene il valore del segnale lag passi indietro nel tempo. Ad esempio IN_lag1
    # contiene IN shiftato di un campione (5 minuti fa), IN_lag2 di due campioni 
    # (10 minuti fa), ecc. Questo trasforma la serie temporale in un problema di regressione
    for stream in streams:
        for lag in range(1, delta_y + 1):
            df[f"{stream}_lag{lag}"] = df[stream].shift(lag)

    # Le prime DELTA_Y righe avranno NaN nei lag (non ci sono campioni passati), quindi vengono eliminate.
    df = df.dropna()
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge feature temporali cicliche e contestuali.

    Feature cicliche (codifica sin/cos per continuità ai bordi del ciclo):
      - Ora del giorno      (periodo 24h)
      - Minuto del giorno   (periodo 288 campioni × 5min = 1 giorno)
      - Giorno della settimana (periodo 7 giorni)
      - Settimana dell'anno (periodo 52 settimane, per stagionalità)

    Feature binarie contestuali:
      - is_weekend:    sabato o domenica
      - is_holiday:    festività italiane
      - is_evening:    fascia 19:00–23:00 (picco streaming)
      - is_dazn_peak:  weekend × sera (picco eventi sportivi DAZN)
      - is_work_hour:  lun-ven 9:00–18:00 (picco VoIP/business)
      - covid_period:  dopo marzo 2020 (shift strutturale VoIP)
    """
    
    # Salva l'indice 'DateTimeIndex in una variabile locale per comodità'
    idx = df.index

    # Calcola l'ora frazionaria del giorno (es. 14:30 → 14.5). Serve per la codifica ciclica dell'ora.
    # Lo stesso vale per le altre quantità
    hour           = idx.hour + idx.minute / 60.0
    minute_of_day  = idx.hour * 12 + idx.minute // 5   # 0..287
    
    # Estrae rispettivamente il giorno della settimana (0=lunedì, 6=domenica) e 
    # il numero della settimana nell'anno secondo lo standard ISO
    day_of_week    = idx.dayofweek                      # 0=lun, 6=dom
    week_of_year   = idx.isocalendar().week.astype(float)

    # Codifica ciclica dell'ora del giorno. Usare sin/cos invece del numero grezzo (0-23) 
    # garantisce che le ore 23 e 0 siano "vicine" numericamente, cosa che un numero 
    # intero non garantirebbe. Lo stesso principio si applica a tutte le coppie sin/cos che seguono.
    # hour_sin e hour_cos codificano già il fatto che i campioni sono 288/giorno
    df["hour_sin"]  = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * hour / 24)

    # Codifica ciclica dell'intervallo di 5 minuti al giorno (288 campioni totali in un giorno).
    df["tod_sin"]   = np.sin(2 * np.pi * minute_of_day / 288)
    df["tod_cos"]   = np.cos(2 * np.pi * minute_of_day / 288)

    # Codifica ciclica del giorno della settimana (ciclo di 7 giorni).
    df["dow_sin"]   = np.sin(2 * np.pi * day_of_week / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * day_of_week / 7)

    # Codifica ciclica della settimana dell'anno (stagionalità annuale, ciclo di 52 settimane)
    df["woy_sin"]   = np.sin(2 * np.pi * week_of_year / 52)
    df["woy_cos"]   = np.cos(2 * np.pi * week_of_year / 52)

    # ── Binarie contestuali ───────────────────
    it_holidays = holidays.Italy()

    # True per sabato (5) e domenica (6), poi convertito in 0/1 con .astype(int).
    df["is_weekend"]   = (day_of_week >= 5).astype(int)

    # idx.normalize() tronca i timestamp alla mezzanotte (rimuove l'ora). Per ogni data, controlla se è una festività italiana e restituisce 1 o 0.
    df["is_holiday"]   = idx.normalize().map(lambda d: int(d in it_holidays))

    # 1 nelle ore serali (19:00–23:00), fascia oraria dove il traffico di streaming Netflix/DAZN è tipicamente massimo.
    df["is_evening"]   = ((idx.hour >= 19) & (idx.hour <= 23)).astype(int)

    # 1 solo quando è sia weekend che sera: identifica il picco caratteristico del traffico DAZN (partite sportive in streaming nel weekend sera).
    df["is_dazn_peak"] = (df["is_weekend"] & df["is_evening"]).astype(int)

    # 1 durante gli orari lavorativi (lunedì-venerdì, 9:00-18:00), fascia dove il traffico VoIP aziendale è predominante.
    df["is_work_hour"] = (
        (day_of_week < 5) & (idx.hour >= 9) & (idx.hour < 18)
    ).astype(int)

    # Periodo Covid19: dal 9 marzo 2020 (primo lockdown Italia)
    covid_start = pd.Timestamp("2020-03-09")
    df["covid_period"] = (idx >= covid_start).astype(int)

    print(f"[features] Aggiunte 14 feature temporali\n")
    return df


def build_feature_matrix(df: pd.DataFrame, streams: list,
                          use_temporal: bool) -> tuple:
    """
    Costruisce la matrice X (input) e il dizionario y (target per stream).

    Args:
        df:           DataFrame con lag e feature temporali
        streams:      lista degli stream da predire
        use_temporal: True = baseline + temporal, False = solo lag (paper)

    Returns:
        X:  np.ndarray (n_samples, n_features)
        y:  dict {stream: np.ndarray (n_samples,)}
        feature_names: list[str]
    """
    # Seleziona tutte le colonne lag presenti nel DataFrame, es. IN_lag1, IN_lag2, OUT_lag1, ecc.
    lag_cols = [c for c in df.columns
                if any(c.startswith(f"{s}_lag") for s in streams)]

    # Se use_temporal=True, prepara la lista delle feature temporali; se False, restituisce una 
    # lista vuota (configurazione baseline del paper).
    temporal_cols = [
        "hour_sin", "hour_cos", "tod_sin",  "tod_cos", "dow_sin",  "dow_cos",  
        "woy_sin", "woy_cos", "is_weekend", "is_holiday", 
        "is_evening", "is_dazn_peak", "is_work_hour", "covid_period"
    ] if use_temporal else []

    # Usa solo le colonne temporali presenti nel df
    temporal_cols = [c for c in temporal_cols if c in df.columns]

    # Concatena le liste di colonne e costruisce la matrice X dei campioni. 
    # .values converte il DataFrame in un array NumPy puro, necessario per scikit-learn
    feature_names = lag_cols + temporal_cols
    X = df[feature_names].values

    # Costruisce un dizionario dove ogni chiave è il nome di uno stream e il valore 
    # è l'array del segnale target (i valori reali da predire).
    y = {stream: df[stream].values for stream in streams}

    label = "lag + temporal" if use_temporal else "solo lag (paper)"
    print(f"[build_features] {label}: "
          f"{X.shape[1]} feature | {X.shape[0]} campioni\n")
    return X, y, feature_names