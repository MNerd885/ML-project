%% Progetto Davide Di Nardo - SDN Traffic Regression
clear all
close all
clc

% PARAMETRI
TRAIN_PATH        = "datasets/SONICATEL_traffic_train.csv";
TEST_PATH         = "datasets/SONICATEL_traffic_test.csv";


% Numero di alberi da generare per Target nell'approccio con le RFs
N_TREES           = [1 2 3 4 5 6 7 8 9 10 30 50 100 200];

MIN_LEAF          = 30;
target_vars = {'VOIP', 'Netflix', 'DAZN'};
features = {'year', 'month', 'day', 'hour', 'min', 'dayweek', 'IN', 'OUT'};

%% Caricamento dei dataset
data_train = readtable(TRAIN_PATH);
data_test = readtable(TEST_PATH);

%% Creazione dei lag utilizzando la lagmatrix 
% Per la creazione del modello autoregressivo per ciascun target devo
% generare una serie temporale che consiste nei valori che il sistema 
% assume agli istanti precedenti, il cui numero è dettato da LAG.

% LAG rappresenta il numero di ritardi da applicare per il modello
% autoregressivo, va inserito dall'utente
LAG = input("Inserire un valore di ritardi (LAG): ");

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
% valori null si interpolano i valori nulli 
data_train = fillmissing(data_train, "linear");
data_test = fillmissing(data_test, "linear");

%% Separazione delle feature dai target
% Per farlo devo estrarre le feature in modo casuale e uniformemente
% distribuito e devo sceglierle in modo da separarle dai target presenti
% nel dataset, considerando progressivamente un numero di LAG più grande

% Usando la function setdiff riesco ad escludere i 3 target, in modo da
% avere solamente le features attualmente disponibili
data_solo_feature = setdiff(data_train.Properties.VariableNames, target_vars);

% Devo in seguito definire quante sceglierne, ad esempio tra 5 e quelle
% disponibili
feat_da_selezionare = randi([5, width(data_train) - length(target_vars)]);

% Estraggo le feature con la function datasample che restituisce k 
% osservazioni del mio dataset di sole feature in modo uniforme senza il
% reinserimento delle feature presenti
scelta_feature = datasample(data_solo_feature,feat_da_selezionare, 'Replace', false);

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
NRMSE_VOIP_RT = sqrt(mean(Y_test.VOIP - VOIP_predicted).^2 ) / (mean(Y_test.VOIP));
disp("NRMSE per il RT del target VOIP: " + num2str(NRMSE_VOIP_RT*100) + "%");

Netflix_predicted = predict(Netflix_RT,X_test);
NRMSE_Netflix_RT = sqrt(mean(Y_test.Netflix - Netflix_predicted).^2) / (mean(Y_test.Netflix));
disp("NRMSE per il RT del target Netflix: " + num2str(NRMSE_Netflix_RT*100) + "%");

DAZN_predicted = predict(DAZN_RT,X_test);
NRMSE_DAZN_RT = sqrt(mean(Y_test.DAZN - DAZN_predicted).^2) / (mean(Y_test.DAZN));
disp("NRMSE per il RT del target DAZN: " + num2str(NRMSE_DAZN_RT*100) + "%");

%% Approccio mediante Random Forests
% Vettori che contengono i NRMSE per ogni foresta con un numero di alberi
% differente
NRMSE_VOIP = zeros(1,size(N_TREES,2));
NRMSE_Netflix = zeros(1,size(N_TREES,2));
NRMSE_DAZN = zeros(1,size(N_TREES,2));
k = 1;

% Definisco il parametro MinLeaf per fitrensemble
t = templateTree('MinLeafSize',MIN_LEAF);

for n = N_TREES
    % Fase di training delle Random Forests
    forestVOIP = fitrensemble(X_train,Y_train.VOIP,"Method","Bag","NumLearningCycles", n,"Learners",t);
    forestNetflix = fitrensemble(X_train,Y_train.Netflix,"Method","Bag","NumLearningCycles", n,"Learners",t);
    forestDAZN = fitrensemble(X_train,Y_train.DAZN,"Method","Bag","NumLearningCycles", n,"Learners",t);

    % Validazione delle perfomance rispetto al NRMSE
    Y_pred_VOIP = predict(forestVOIP, X_test);
    NRMSE_VOIP(:,k) = sqrt(mean(Y_test.VOIP - Y_pred_VOIP).^2) / (mean(Y_test.VOIP));
    NRMSE_VOIP(:,k) = NRMSE_VOIP(:,k)*100;
    disp("NRMSE RF VOIP con " + num2str(n) + " alberi/o: " + num2str(NRMSE_VOIP(:,k)) + "%");

    Y_pred_Netflix = predict(forestNetflix, X_test);
    NRMSE_Netflix(:,k) = sqrt(mean(Y_test.Netflix - Y_pred_Netflix).^2) / (mean(Y_test.Netflix));
    NRMSE_Netflix(:,k) = NRMSE_Netflix(:,k)*100;
    disp("NRMSE RF Netflix con " + num2str(n) + " alberi/o: " + num2str(NRMSE_Netflix(:,k)) + "%");
    

    Y_pred_DAZN = predict(forestDAZN, X_test);
    NRMSE_DAZN(:,k) = sqrt(mean(Y_test.DAZN - Y_pred_DAZN).^2) / (mean(Y_test.DAZN));
    NRMSE_DAZN(:,k) = NRMSE_DAZN(:,k)*100;
    disp("NRMSE RF DAZN con " + num2str(n) + " alberi/o: " + num2str(NRMSE_DAZN(:,k)) + "%");

    k = k+1;
end

%% Plot NRMSE al crescere degli alberi

figure()
subplot(3,1,1)
bar(NRMSE_VOIP); grid on;
xlabel('Numero di alberi cresciuti');
xticklabels({"1","2","3","4","5","6","7","8","9","10","30","50","100","200"});
ylabel('NRMSE VOIP[%]');
title("NRMSE del target VOIP al variare del numero di alberi");

subplot(3,1,2)
bar(NRMSE_Netflix); grid on;
xlabel('Number of grown trees');
xticklabels({"1","2","3","4","5","6","7","8","9","10","30","50","100","200"});
ylabel('NRMSE Netflix[%]')
title("NRMSE del target Netflix al variare del numero di alberi");

subplot(3,1,3)
bar(NRMSE_DAZN); grid on;
xticklabels({"1","2","3","4","5","6","7","8","9","10","30","50","100","200"});
xlabel('Numero di alberi cresciuti');
ylabel('NRMSE DAZN[%]');
title("NRMSE del target DAZN al variare del numero di alberi");

%% Plot di confronto tra i valori predetti di ciascun target con i valori reali

figure()

% Plot VOIP
subplot(3,1,1)
hold on;
plot(Y_test.VOIP,Y_pred_VOIP,'bo','LineWidth',2); % Valori predetti
plot(Y_test.VOIP,Y_test.VOIP,'r-','LineWidth',2); % Valori reali (y = x)
xlabel('Valori Reali VOIP');
ylabel('Previsioni VOIP');
title("Confronto tra previsioni vs valori reali del target VOIP (NRMSE=" + num2str(NRMSE_VOIP(end))+ " %)");
legend('Yhat_{VOIP}', 'Ytest_{VOIP}');
grid on;
hold off;

% Plot Netflix
subplot(3,1,2)
hold on;
plot(Y_test.Netflix,Y_pred_Netflix,'go','LineWidth',2); % Valori predetti
plot(Y_test.Netflix,Y_test.Netflix,'r-','LineWidth',2); % Valori reali (y = x)
xlabel('Valori Reali Netflix');
ylabel('Previsioni Netflix');
title("Confronto tra previsioni vs valori reali del target Netflix (NRMSE=" + num2str(NRMSE_Netflix(end))+ " %)");
legend('Yhat_{Netflix}', 'Ytest_{Netflix}');
grid on;
hold off;

% Plot DAZN
subplot(3,1,3)
hold on;
plot(Y_test.DAZN,Y_pred_DAZN,'yo','LineWidth',2); % Valori predetti
plot(Y_test.DAZN,Y_test.DAZN,'r-','LineWidth',2); % Valori reali (y = x)
xlabel('Valori Reali DAZN');
ylabel('Previsioni DAZN');
title("Confronto tra previsioni vs valori reali del target DAZN (NRMSE=" + num2str(NRMSE_DAZN(end))+ " %)");
legend('Yhat_{DAZN}', 'Ytest_{DAZN}');
grid on;
hold off;

%% Importanza delle feature

impVOIP = oobPermutedPredictorImportance(forestVOIP);
impNetflix = oobPermutedPredictorImportance(forestNetflix);
impDAZN = oobPermutedPredictorImportance(forestDAZN);

%% Plot dell'importanza di ciascuan feature usata nella predizione dei 3 target 
figure()
bar(scelta_feature,impVOIP);
ylabel("Importanza delle feature")
xlabel("Nome delle feature")
title("Confronto tra il contributo che ciscuna feature dà alla riduzione dell'errore")

figure()
bar(scelta_feature,impNetflix);
ylabel("Importanza delle feature")
xlabel("Nome delle feature")
title("Confronto tra il contributo che ciscuna feature dà alla riduzione dell'errore")

figure()
bar(scelta_feature,impDAZN);
ylabel("Importanza delle feature")
xlabel("Nome delle feature")
title("Confronto tra il contributo che ciscuna feature dà alla riduzione dell'errore")

%% Selezione delle feature con importanza molto grande

% Per selezionare le feature che per importanza contribuiscono alla
% riduzione del NRMSE si utilizza l'algoritmo Boruta 
% (fonte: https://www.diariodiunanalista.it/posts/selezione-delle-feature-con-boruta/)
% il quale consiste nell'inserire una colonna mai vista (detta "shadow", se
% ne inserisce una sola per semplicità) nel dataset orginale la quale è una
% una feature sintestica generata randomicamente.
% In seguito si allena una nuova foresta per ciascuno dei target e
% poi si calcola lo score (NRMSE) e l'importanza delle feature, se le
% feature hanno un'importanza maggiore rispetto alla feature shadow allora
% viene scelta la feature con importanza maggiore rispetto a quella shadow.
% Dopo tutto ciò si allena il modello ridotto e si confronta il NRMSE del 
% modello orginale e quello ridotto.

% Inserisco una colonna randomizzata nel dataset di training originale
rng(45) % Riproducibilità dell'esprimento
X_shadow = X_train;
X_shadow = addvars(X_shadow, rand(size(X_train,1),1), 'NewVariableNames','feat_sintetica');

% Training sul dataset shadow
forestVOIP_shadow = fitrensemble(X_shadow,Y_train.VOIP,"Method","Bag","NumLearningCycles", n,"Learners",t);
forestNetflix_shadow = fitrensemble(X_shadow,Y_train.Netflix,"Method","Bag","NumLearningCycles", n,"Learners",t);
forestDAZN_shadow = fitrensemble(X_shadow,Y_train.DAZN,"Method","Bag","NumLearningCycles", n,"Learners",t);

% Calcolo l'importanza delle feature di X_shadow in modo da poter poi
% confrontarle tutte con la feature sintetica
imp_VOIP_shadow = oobPermutedPredictorImportance(forestVOIP_shadow);
imp_Netflix_shadow = oobPermutedPredictorImportance(forestNetflix_shadow);
imp_DAZN_shadow = oobPermutedPredictorImportance(forestDAZN_shadow);

% Confronto delle feature con quella sintetica
% Estraggo l'importanza delle feature sintetiche
imp_sint_VOIP = imp_VOIP_shadow(end);
imp_sint_Netflix = imp_Netflix_shadow(end);
imp_sint_DAZN = imp_DAZN_shadow(end);

% Seleziono le feature del dataset di partenza
feat_imp_VOIP = imp_VOIP_shadow(1:end-1);
feat_imp_DAZN = imp_Netflix_shadow(1:end-1);
feat_imp_Netflix = imp_DAZN_shadow(1:end-1);

% Selezione delle sole feature che hanno importanza maggiore di quella
% sintetica
sel_id_VOIP = feat_imp_VOIP > imp_sint_VOIP;
sel_id_Netflix = feat_imp_Netflix > imp_sint_Netflix;
sel_id_DAZN = feat_imp_DAZN > imp_sint_DAZN;

%% Confronto tra il modello originale e ridotto
X_train_red_VOIP = X_train(:,sel_id_VOIP);
X_train_red_Netflix = X_train(:,sel_id_Netflix);
X_train_red_DAZN = X_train(:,sel_id_DAZN);

% Training sul dataset ridotto
forestVOIP_red = fitrensemble(X_train_red_VOIP,Y_train.VOIP,"Method","Bag","NumLearningCycles", n,"Learners",t);
forestNetflix_red = fitrensemble(X_train_red_Netflix,Y_train.Netflix,"Method","Bag","NumLearningCycles", n,"Learners",t);
forestDAZN_red = fitrensemble(X_train_red_DAZN,Y_train.DAZN,"Method","Bag","NumLearningCycles", n,"Learners",t);

% Validazione delle perfomance rispetto al NRMSE
Yhat_VOIP_red = predict(forestVOIP_red, X_test);
NRMSE_VOIP_red = sqrt(mean(Y_test.VOIP - Yhat_VOIP_red).^2) / (mean(Y_test.VOIP));
NRMSE_VOIP_red = NRMSE_VOIP_red*100;
fprintf('--- %s ---\n', target_vars{1});
fprintf('Feature selezionate: %d/%d\n', sum(sel_id_VOIP), numel(sel_id_VOIP));
fprintf('NRMSE completo : %.4f\n', NRMSE_VOIP(:,end));
fprintf('NRMSE ridotto  : %.4f\n', NRMSE_VOIP_red);

Yhat_Netflix_red = predict(forestNetflix_red, X_test);
NRMSE_Netflix_red = sqrt(mean(Y_test.Netflix - Yhat_Netflix_red).^2) / (mean(Y_test.Netflix));
NRMSE_Netflix_red = NRMSE_Netflix_red*100;
fprintf('--- %s ---\n', target_vars{2});
fprintf('Feature selezionate: %d/%d\n', sum(sel_id_Netflix), numel(sel_id_Netflix));
fprintf('NRMSE completo : %.4f\n', NRMSE_Netflix(:,end));
fprintf('NRMSE ridotto  : %.4f\n', NRMSE_Netflix_red);

Yhat_DAZN_red = predict(forestDAZN_red, X_test);
NRMSE_DAZN_red = sqrt(mean(Y_test.DAZN - Yhat_DAZN_red).^2) / (mean(Y_test.DAZN));
NRMSE_DAZN_red = NRMSE_DAZN_red*100;
fprintf('--- %s ---\n', target_vars{3});
fprintf('Feature selezionate: %d/%d\n', sum(sel_id_DAZN), numel(sel_id_DAZN));
fprintf('NRMSE completo : %.4f\n', NRMSE_DAZN(:,end));
fprintf('NRMSE ridotto  : %.4f\n', NRMSE_DAZN_red);