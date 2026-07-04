%% Progetto Davide Di Nardo - SDN Traffic Regression
clear all
close all
clc

% PARAMETRI

TRAIN_PATH        = "..\datasets\SONICATEL_traffic_train.csv";
TEST_PATH         = "..\datasets\SONICATEL_traffic_test.csv";

% LAG rappresenta il numero di ritardi da applicare per il modello
% autoregressivo, in questo caso 
LAG               = 15; 

% Numero di alberi da generare per Target nell'approccio con le RFs
N_TREES           = [1 2 3 4 5 6 7 8 9 10 30 50 100 200];

MIN_LEAF          = 15;
target_vars = {'VOIP', 'Netflix', 'DAZN'};
features = {'year', 'month', 'day', 'hour', 'min', 'dayweek', 'IN', 'OUT'};

%% Caricamento dei dataset
data_train = readtable(TRAIN_PATH);
data_test = readtable(TEST_PATH);

%% Creazione dei lag utilizzando la lagmatrix 
% Per la creazione del modello autoregressivo per ciascun target devo
% generare una serie temporale che consiste nei valori che il sistema 
% assume agli istanti precedenti, il cui numero è dettato da LAG.
for i = 1:3
    
    % In questo modo genero ad ogni iterazione una matrice lag sia per i
    % dati di train che test, associata a ciascuno dei tre target
    lag_data_train = lagmatrix(data_train.(target_vars{i}), 1:LAG);
    lag_data_test = lagmatrix(data_test.(target_vars{i}),1:LAG);

    % Qui aggiungo i lag precedentemente generati come nuove colonne
    %  dei due dataset che ho caricato precedentemente per ottenere il
    %  modello autoregressivo
    for j = 1:LAG
        data_train.(target_vars{i} + "_lag_" + j) = lag_data_train(:,j);
        data_test.(target_vars{i} + "_lag_" + j) = lag_data_test(:,j);
    end
end

%% Pulizia dei dataset 
% L'obiettivo è di eliminare i valori NaN presenti introdotti dai lag
% inseriti, ciò accade proprio perché ritardando i valori del target nel
% tempo non si hanno informazioni precedenti riguardo l'evoluzione del
% modello.
% Per quanto riguarda la pulizia del dataset che concerne la gestione dei
% valori null si prova sia ad interpolare i valori nulli e sia a rimuoverli
% completamente

data_train = fillmissing(data_train, "linear");
data_test = fillmissing(data_test, "linear");

%% Separazione delle feature dai target
% Per farlo devo estrarre le feature in modo casuale e uniformemente
% distribuito e devo sceglierle in modo da separarle dai target presenti
% nel dataset, considerando progressivamente un numero di LAG più grande

% Usando la function setdiff riesco ad escludere i 3 target, in modo da
% avere solamente le features attualmente disponibili
data_solo_feature = setdiff(data_train.Properties.VariableNames, target_vars, 'stable');

% Devo in seguito definire quante sceglierne, ad esempio tra 4 e quelle
% disponibili
feat_da_selezionare = randi([5, width(data_train) - length(target_vars)]);

% Estraggo le feature con la function datasample che restituisce k 
% osservazioni del mio dataset di sole feature in modo uniforme senza il
% rimpiazzamento delle feature presenti
scelta_feature = datasample(data_solo_feature,feat_da_selezionare,'Replace', false);

% Qui separo effettivamente le feature dai target che voglio predire

X_train = data_train(:,scelta_feature);
X_test = data_test(:,scelta_feature);
Y_train = data_train(:,target_vars);
Y_test = data_test(:,target_vars);

%% Train degli alberi e plot

VOIP_RT = fitrtree(X_train, Y_train.VOIP , 'MinLeafSize', MIN_LEAF);
Netflix_RT = fitrtree(X_train, Y_train.Netflix , 'MinLeafSize', MIN_LEAF);
DAZN_RT = fitrtree(X_train, Y_train.DAZN , 'MinLeafSize', MIN_LEAF);

view(VOIP_RT, 'Mode','graph');
view(Netflix_RT, 'Mode','graph');
view(DAZN_RT, 'Mode','graph');

%% Testing e valutazione performance
% Per valutare performance del mio predittore vedo come questo predice i 3
% target con un dataset di test e per ciascuno di essi valuto il NRMSE di
% ciascun RT

VOIP_predicted = predict(VOIP_RT,X_test);
NRMSE_VOIP_RT = sqrt(mean(Y_test.VOIP - VOIP_predicted).^2) / (mean(Y_test.VOIP));
disp("NRMSE per il RT del target VOIP: " + num2str(NRMSE_VOIP_RT*100) + "%");

Netflix_predicted = predict(Netflix_RT,X_test);
NRMSE_Netflix_RT = sqrt(mean(Y_test.Netflix - Netflix_predicted).^2) / (mean(Y_test.Netflix));
disp("NRMSE per il RT del target Netflix: " + num2str(NRMSE_Netflix_RT*100) + "%");

DAZN_predicted = predict(DAZN_RT,X_test);
NRMSE_DAZN_RT = sqrt(mean(Y_test.DAZN - DAZN_predicted).^2) / (mean(Y_test.DAZN));
disp("NRMSE per il RT del target DAZN: " + num2str(NRMSE_DAZN_RT*100) + "%");

%% Approccio mediante Random Forests

for n = N_TREES
    % Fase di training delle Random Forests
    forestVOIP = TreeBagger(n, X_train, Y_train.VOIP, 'Method', 'regression', 'MinLeafSize', MIN_LEAF);
    forestNetflix = TreeBagger(n, X_train, Y_train.Netflix, 'Method', 'regression', 'MinLeafSize', MIN_LEAF);
    forestDAZN = TreeBagger(n, X_train, Y_train.DAZN, 'Method', 'regression', 'MinLeafSize', MIN_LEAF);


end