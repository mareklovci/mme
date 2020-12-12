%% KMA/MME Linearization

% Clear Sequence
clear all, close all %#ok<CLALL>

% Wait 0.01s (sometimes, clear all does not delete everything)
pause(0.01)

%% Load Data

load data

y = data(:, 2);
x = data(:, 1);
n = length(x);

x_min = min(x);
x_max = max(x);

%% Naivni odhady

% graf zakladniho vztahu
figure
plot(x, y, 'or'), hold on
title('Ziskana pozorovani')
xlabel('x'), ylabel('y')

% nelinearni funkce - spravne
funkce = @(beta, x) -log(beta(1)) + log(log(x/beta(2)) + 1);

%pocatecni nastaveni parametru beta
beta0 = [1 1];

% odhad parametru nelinearni regrese
% b_nlin = koeficienty, sigma = kov. matice koeficientu, mse = rezisualni roztpyl
[b_nlin, resid, J, sigma, mse] = nlinfit(x, y, funkce, beta0);

% 95% intervaly spolehlivosti pro parametry beta
ci = nlparci(b_nlin, resid, 'covariance', sigma);

% graf
xfit = linspace(x_max, x_min);
plot(xfit, funkce(b_nlin, xfit))

% 95% predikcni interval (interval spolehlivosti)
[ypred, delta] = nlpredci(funkce, xfit, b_nlin, resid, 'covariance', sigma);

% horni a dolni mezni intervaly
dolni = ypred - delta;
horni = ypred + delta;

% vykresleni
plot(xfit, [dolni; horni], 'r--')

%% nastroj

nlintool(x, y, funkce, beta0, 0.05)
