import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ─── CONFIGURAZIONE ───────────────────────────────────────────────────────
TRAIN_PATH = 'datasets/SONICATEL_traffic_train.csv'
TEST_PATH  = 'datasets//SONICATEL_traffic_test.csv'
DELTA_Y    = 5
N_HORIZON  = 10
N_TREES    = 30
MIN_LEAF   = 5
STREAMS    = ['IN', 'OUT', 'VOIP', 'Netflix', 'DAZN']
STREAMS    = ['IN', 'OUT', 'VOIP']

# ─── CARICAMENTO ──────────────────────────────────────────────────────────
Ttr = pd.read_csv(TRAIN_PATH)
Tte = pd.read_csv(TEST_PATH)

for df in [Ttr, Tte]:
    df['timestamp'] = pd.to_datetime(df[['year','month','day','hour']].rename(columns={'hour':'hour'}).assign(minute=df['min']))
    df.set_index('timestamp', inplace=True)

Ytr = Ttr[STREAMS].clip(lower=0)
Yte = Tte[STREAMS].clip(lower=0)
ts_train = Ttr.index
ts_test  = Tte.index

print(f'Train: {len(Ytr)} campioni | Test: {len(Yte)} campioni')

# ─── LAG FEATURES ─────────────────────────────────────────────────────────
def make_lags(Y, delta_y):
    return pd.concat([Y.shift(d).add_suffix(f'_lag{d}') for d in range(1, delta_y+1)], axis=1).iloc[delta_y:]

Xtr_lag = make_lags(Ytr, DELTA_Y)
Xte_lag = make_lags(Yte, DELTA_Y)
Ytr = Ytr.iloc[DELTA_Y:];  ts_train = ts_train[DELTA_Y:]
Yte = Yte.iloc[DELTA_Y:];  ts_test  = ts_test[DELTA_Y:]

# ─── FEATURE TEMPORALI ────────────────────────────────────────────────────
def make_temporal(ts):
    h   = ts.hour + ts.minute / 60
    tod = ts.hour * 12 + ts.minute // 5
    dow = ts.dayofweek
    woy = ts.isocalendar().week.astype(float)
    return pd.DataFrame({
        'hour_sin':     np.sin(2*np.pi*h/24),
        'hour_cos':     np.cos(2*np.pi*h/24),
        'tod_sin':      np.sin(2*np.pi*tod/288),
        'tod_cos':      np.cos(2*np.pi*tod/288),
        'dow_sin':      np.sin(2*np.pi*dow/7),
        'dow_cos':      np.cos(2*np.pi*dow/7),
        'woy_sin':      np.sin(2*np.pi*woy/52),
        'woy_cos':      np.cos(2*np.pi*woy/52),
        'is_weekend':   (dow >= 5).astype(int),
        'is_evening':   ((ts.hour >= 19) & (ts.hour <= 23)).astype(int),
        'is_dazn_peak': ((dow >= 5) & (ts.hour >= 19) & (ts.hour <= 23)).astype(int),
        'is_work_hour': ((dow < 5) & (ts.hour >= 9) & (ts.hour < 18)).astype(int),
        'covid_period': (ts >= pd.Timestamp('2020-03-09')).astype(int),
    }, index=ts)

Xtr_temp = make_temporal(ts_train)
Xte_temp = make_temporal(ts_test)

Xtr_b = Xtr_lag.values;                        Xte_b = Xte_lag.values
Xtr_e = np.hstack([Xtr_lag, Xtr_temp]);        Xte_e = np.hstack([Xte_lag, Xte_temp])
Ytr   = Ytr.values;                            Yte   = Yte.values

# ─── TRAINING E VALUTAZIONE ───────────────────────────────────────────────
nrmse_base = np.zeros((len(STREAMS), N_HORIZON))
nrmse_enri = np.zeros((len(STREAMS), N_HORIZON))

for mode, (Xtr, Xte, label) in enumerate([(Xtr_b, Xte_b, 'BASELINE'), (Xtr_e, Xte_e, 'ARRICCHITO')]):
    print(f'\n[{label}]')
    for s, stream in enumerate(STREAMS):
        for j in range(N_HORIZON):
            n  = len(Xtr) - (j+1)
            rf = RandomForestRegressor(n_estimators=N_TREES, min_samples_leaf=MIN_LEAF, n_jobs=-1, random_state=42)
            rf.fit(Xtr[:n], Ytr[j+1:j+1+n, s])
            yp = rf.predict(Xte)
            yt = Yte[j+1:len(yp)+j+1, s]
            yp = yp[:len(yt)]
            err = np.sqrt(mean_squared_error(yt, yp)) / (yt.max() - yt.min()) * 100
            if mode == 0: nrmse_base[s, j] = err
            else:         nrmse_enri[s, j] = err
        e = nrmse_base if mode == 0 else nrmse_enri
        print(f'  {stream}: N=1 → {e[s,0]:.2f}%  N=10 → {e[s,-1]:.2f}%')

# ─── PLOT ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(STREAMS), 1, figsize=(9, 3*len(STREAMS)), sharex=True)
for ax, s, stream in zip(axes, range(len(STREAMS)), STREAMS):
    ax.plot(range(1, N_HORIZON+1), nrmse_base[s], 'o--r', lw=1.5, label='RF (solo lag)')
    ax.plot(range(1, N_HORIZON+1), nrmse_enri[s], 's-b', lw=2.0, label='RF + temporali')
    ax.set_ylabel('NRMSE (%)'); ax.set_title(stream); ax.legend(fontsize=8); ax.grid(True)
axes[-1].set_xlabel('Prediction Horizon N')
plt.suptitle('Confronto NRMSE: Baseline vs Feature Temporali', fontsize=13)
plt.tight_layout(); plt.show()

# ─── RIEPILOGO ────────────────────────────────────────────────────────────
print(f'\n{"Stream":<15} {"Baseline":>10} {"Arricchito":>12} {"Delta":>8}')
print('-' * 48)
for s, stream in enumerate(STREAMS):
    b, e = nrmse_base[s,0], nrmse_enri[s,0]
    print(f'{stream:<15} {b:>10.2f}% {e:>11.2f}% {e-b:>+8.2f}%')