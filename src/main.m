clear all
close all
clc

TRAIN_PATH = "..\datasets\SONICATEL_traffic_train.csv";
TEST_PATH  = "..\datasets\SONICATEL_traffic_test.csv";
DELTA_Y    = 5;
N_HORIZON  = 10;
N_TREES    = 30;
MIN_LEAF   = 5;
STR2    = ['IN', 'OUT', 'VOIP', 'Netflix', 'DAZN'];
STR1    = ['IN', 'OUT', 'VOIP'];

%% Caricamento dataset e preprocessing

data = readtable("..\datasets\SONICATEL_traffic_train.csv");

