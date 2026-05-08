# ─────────────────────────────────────────────
# 1. CONFIGURAZIONE E COSTANTI
# ─────────────────────────────────────────────

# Parametri del paper (Tabella I)
DELTA_Y     = 20         # ordine del lag
N_HORIZON   = 10         # orizzonte predittivo (10 step = 50 minuti)
N_TREES     = 30         # numero di alberi per RF
MIN_LEAF    = 5          # minimo campioni per foglia (RF)
SAMPLING    = "5min"     # intervallo di campionamento che rappresenta 5 minuti di traffico

# Stream di traffico disponibili
# Il dataset Sonicatel usa questi nomi di colonna esatti:
STREAMS_FA1 = ["IN", "OUT", "VOIP"]
STREAMS_FA2 = ["IN", "OUT", "VOIP", "Netflix", "DAZN"]

# Split train/test (seguiamo il paper)
# FA1: train ~91 giorni, test ~66 giorni
# FA2 Netflix: train ~55 giorni, test ~2 giorni
# FA2 DAZN:   train ~29 giorni, test ~2 giorni (weekend)
TRAIN_RATIO = 0.60       # usato come fallback se non si specificano i giorni

# Dataset di training e testing e numero dell'access point
TRAIN_PATH   = "../datasets/SONICATEL_traffic_train.csv" 
TEST_PATH    = "../datasets/SONICATEL_traffic_test.csv"   
ACCESS_POINT = "FA1"                           
#ACCESS_POINT = "FA2" 