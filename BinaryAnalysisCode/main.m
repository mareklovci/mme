%% KMA/MME Logit & Probit

% Clear Sequence
clear all, close all %#ok<CLALL>

% Wait 0.01s (sometimes, clear all does not delete everything)
pause(0.01)

%% Load Data
% Price of Monet Paintings
% (price, height, width, signed, picture = id, house)

input = importdata('TableF4-1.csv');

y = input.data(:, 4); % signed

x1 = input.data(:, 1); % price
x2 = input.data(:, 2) .* input.data(:, 3); % height * width = area [inch^2]
x3 = input.data(:, 6); % house

X = [x1 x2 x3];

% Output Table format text
formatSpec = '[deviation] probit: %3.4f, logit: %3.4f, LSE: %3.4f \n';

%% Plot 'Em All

figure
subplot(2, 2, 1), hold on
plot(x1, y, 'o'), grid on
xlabel('Price'), ylabel('Signed');

subplot(2, 2, 2)
plot(x2, y, 'o'), grid on
xlabel('Area'), ylabel('Signed');

subplot(2, 2, [3 4])
plot(x3, y, 'o'), grid on
xlabel('House'), ylabel('Signed');

hold off

saveas(gcf, 'fig1', 'epsc')

%% Count Data

formatSpec10 = 'Number of zeros: %d, ones: %d \n';

% Count Zeros & Ones
fprintf(formatSpec10, nnz(~y), nnz(y))

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
xlabel('Price'), ylabel('Signed');

% pomocný prostor pro vykreslení spojité funkce
xx = linspace(floor(min(x1)), ceil(max(x1)));

% Probit (glmfit pøidá jednotkový sloupec automaticky)
[x1B_probit, x1Dev_probit] = glmfit(x1, y, 'binomial', 'link', 'probit');

% x1B_probit: Odhad koeficientù
% x1Dev_probit: Reziduální souèet ètvercù
% x1Stats_probit: Souhrnné statistiky

% Plot Probit
plot(xx, glmval(x1B_probit, xx, 'probit'), '-r')

% Interpretace: pøi maximálním poètu bodù v datech máme stále jen 50% šanci
% na pøijetí

% Logit
[x1B_logit, x1Dev_logit] = glmfit(x1, y, 'binomial', 'link', 'logit');

% Plot Logit
plot(xx, glmval(x1B_logit, xx, 'logit'), '-g')

% Klasická metoda nejmenších ètvercù
[x1B_mnc, x1Dev_mnc] = glmfit(x1, y, 'binomial', 'link', 'identity');
plot(xx, glmval(x1B_mnc, xx, 'identity'), '-b'),
legend('data', 'probit', 'logit', 'LSE'), hold off

% Print results
fprintf(formatSpec, x1Dev_probit, x1Dev_logit, x1Dev_mnc)

% Interpretace: Rozdíly odchylek jsou tak malé, že v tomto pøípadì závisí
% na nás, kterou metodu zvolíme, jinak se volí logit/probit

saveas(gcf, 'fig2', 'epsc')

%% Logit & Probit Regrese X2

figure
plot(x2, y, 'o'), grid on, hold on
xlabel('Area'), ylabel('Signed');

% pomocný prostor pro vykreslení spojité funkce
xx = linspace(floor(min(x2)), ceil(max(x2)));

% Probit (glmfit pøidá jednotkový sloupec automaticky)
[x2B_probit, x2Dev_probit] = glmfit(x2, y, 'binomial', 'link', 'probit');

% Plot Probit
plot(xx, glmval(x2B_probit, xx, 'probit'), '-r')

% Logit
[x2B_logit, x2Dev_logit] = glmfit(x2, y, 'binomial', 'link', 'logit');

% Plot Logit
plot(xx, glmval(x2B_logit, xx, 'logit'), '-g')

% Klasická metoda nejmenších ètvercù
[x2B_mnc, x2Dev_mnc] = glmfit(x2, y, 'binomial', 'link', 'identity');
plot(xx, glmval(x2B_mnc, xx, 'identity'), '-b'),
legend('data', 'probit', 'logit', 'LSE'), hold off

% Print results
fprintf(formatSpec, x2Dev_probit, x2Dev_logit, x2Dev_mnc)

saveas(gcf, 'fig3', 'epsc')

%% Logit & Probit Regrese X3

figure
plot(x3, y, 'o'), grid on, hold on
xlabel('House'), ylabel('Signed');

% pomocný prostor pro vykreslení spojité funkce
xx = linspace(floor(min(x3)), ceil(max(x3)));

% Probit (glmfit pøidá jednotkový sloupec automaticky)
[x3B_probit, x3Dev_probit] = glmfit(x3, y, 'binomial', 'link', 'probit');

% Plot Probit
plot(xx, glmval(x3B_probit, xx, 'probit'), '-r')

% Logit
[x3B_logit, x3Dev_logit] = glmfit(x3, y, 'binomial', 'link', 'logit');

% Plot Logit
plot(xx, glmval(x3B_logit, xx, 'logit'), '-g')

% Klasická metoda nejmenších ètvercù
[x3B_mnc, x3Dev_mnc] = glmfit(x3, y, 'binomial', 'link', 'identity');
plot(xx, glmval(x3B_mnc, xx, 'identity'), '-b'),
legend('data', 'probit', 'logit', 'LSE'), hold off

% Print results
fprintf(formatSpec, x3Dev_probit, x3Dev_logit, x3Dev_mnc)

saveas(gcf, 'fig4', 'epsc')

%% Model v závislosti na X

% Probit (glmfit pøidá jednotkový sloupec automaticky)
[~, xDev_probit] = glmfit(X, y, 'binomial', 'link', 'probit');

% Logit
[~, xDev_logit] = glmfit(X, y, 'binomial', 'link', 'logit');

% Klasická metoda nejmenších ètvercù
[~, xDev_mnc] = glmfit(X, y, 'binomial', 'link', 'identity');

% Print results
fprintf(formatSpec, xDev_probit, xDev_logit, xDev_mnc)
