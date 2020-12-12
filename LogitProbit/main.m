%% KMA/MME Logit & Probit

% Clear Sequence
clear all, close all %#ok<CLALL>

% Wait 0.01s (sometimes, clear all does not delete everything)
pause(0.01)

%% Load Data

% ADMIT - binární pøijat/nepøijat
% GRE (Graduate Rocord Exam Scores) výsledky závìreèného hodnocení na SŠ
% TOPNOTCH - Patøí škola mezi TOP
% GPA (Grade Point Average)

% Úkol: sestavit model popisující šance pøijetí studenta na VŠ a odhadnìte
% parametry modelu pomocí: probit, logit a metodou minimalizace kvadrátù
% odchylek

input = importdata('data04_01.txt');

y = input.data(:, 1);
X = input.data(:, 2:end);

x1 = input.data(:, 2);
x2 = input.data(:, 3);
x3 = input.data(:, 4);

%% Plot 'Em All

figure
plot(x1, y, 'o'), grid on
xlabel('GRE'), ylabel('ADMIT');

figure
plot(x2, y, 'o'), grid on
xlabel('TOPNOTCH'), ylabel('ADMIT');

figure
plot(x3, y, 'o'), grid on
xlabel('GPA'), ylabel('ADMIT');

% Z pohledu na data zjistíme, že 2. promìnná je nic neøíkající, tudíž by
% bylo vhodné ji z modelu vyøadit

%% Lineární regresní model

stats = regstats(y, X, 'linear');

% Testování vhodnosti modelu pomocí F-testu
% H0: Všechny beta = 0, HA: alespoò 1 je rùzná od 0
F_statistika = stats.fstat.pval; 
disp(F_statistika)

% Výsledek: menší než alpha => zamítám H0, tudíž model má smysl

% Test, zda jsou jednotlivé koeficienty vhodné do modelu, pomocí t-testu
% H0: koeficient = 0
t_statistika = stats.tstat.pval;
disp(t_statistika)

% Výsledek: vektor p-hodnot každého koef. beta modelu, hodnoty vyšší než 5%
% vedou k zamítnutí H0 hypotézy a znaèí statistickou insignifikanci daného
% parametru. V tomto pøípadì je 3. hodnota vyšší než hladina alfa, tudíž
% bychom ji mìli z modelu vyøadit

%% Logit & Probit Regrese X1

figure
plot(x1, y, 'o'), grid on, hold on
xlabel('GRE'), ylabel('ADMIT');

% pomocný prostor pro vykreslení spojité funkce
xx = linspace(floor(min(x1)), ceil(max(x1)));

% Probit (glmfit pøidá jednotkový sloupec automaticky)
[x1B_probit, x1Dev_probit, x1Stats_probit] = glmfit(x1, y, 'binomial', 'link', 'probit');

% x1B_probit: Odhad koeficientù
% x1Dev_probit: Rezisuální souèet ètvercù
% x1Stats_probit: Souhrnné statistiky

% Plot Probit
plot(xx, glmval(x1B_probit, xx, 'probit'), '-r')

% Interpretace: pøi maximálním poètu bodù v datech máme stále jen 50% šanci
% na pøijetí

% Logit
[x1B_logit, x1Dev_logit, x1Stats_logit] = glmfit(x1, y, 'binomial', 'link', 'logit');

% Plot Logit
plot(xx, glmval(x1B_logit, xx, 'logit'), '-g')

% Klasická metoda nejmenších ètvercù
[x1B_mnc, x1Dev_mnc, x1Stats_mnc] = glmfit(x1, y, 'binomial', 'link', 'identity');
plot(xx, glmval(x1B_mnc, xx, 'identity'), '-b'), hold off

disp(x1Dev_probit)
disp(x1Dev_logit)
disp(x1Dev_mnc)

% Interpretace: Rozdíly odchylek jsou tak malé, že v tomto pøípadì závisí
% na nás, kterou metodu zvolíme, jinak se volí logit/probit

%% Logit & Probit Regrese X3

figure
plot(x3, y, 'o'), grid on, hold on
xlabel('GPA'), ylabel('ADMIT');

% pomocný prostor pro vykreslení spojité funkce
xx = linspace(floor(min(x3)), ceil(max(x3)));

% Probit (glmfit pøidá jednotkový sloupec automaticky)
[x3B_probit, x3Dev_probit, x3Stats_probit] = glmfit(x3, y, 'binomial', 'link', 'probit');

% Plot Probit
plot(xx, glmval(x3B_probit, xx, 'probit'), '-r')

% Logit
[x3B_logit, x3Dev_logit, x3Stats_logit] = glmfit(x3, y, 'binomial', 'link', 'logit');

% Plot Logit
plot(xx, glmval(x3B_logit, xx, 'logit'), '-g')

% Klasická metoda nejmenších ètvercù
[x3B_mnc, x3Dev_mnc, x3Stats_mnc] = glmfit(x3, y, 'binomial', 'link', 'identity');
plot(xx, glmval(x3B_mnc, xx, 'identity'), '-b'), hold off

disp(x3Dev_probit)
disp(x3Dev_logit)
disp(x3Dev_mnc)

%% Model v závislosti na x1 a x3

Xnew = [x1, x3];

% Probit (glmfit pøidá jednotkový sloupec automaticky)
[x13B_probit, x13Dev_probit, x13Stats_probit] = glmfit(Xnew, y, 'binomial', 'link', 'probit');

% Logit
[x13B_logit, x13Dev_logit, x13Stats_logit] = glmfit(Xnew, y, 'binomial', 'link', 'logit');

% Klasická metoda nejmenších ètvercù
[x13B_mnc, x13Dev_mnc, x13Stats_mnc] = glmfit(Xnew, y, 'binomial', 'link', 'identity');

disp(x13Dev_probit)
disp(x13Dev_logit)
disp(x13Dev_mnc)