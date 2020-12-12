%% KMA/MME Linearization & Confidence Intervals

% Clear Sequence
clear all, close all %#ok<CLALL>

% Wait 0.01s (sometimes, clear all does not delete everything)
pause(0.01)

%% Load Data

data = xlsread('data05.xlsx', 'data4');
y = data(:, 1); x = data(:, 2);
n = length(x);

%% Data Generator

generator = @(beta, x) x ./ (beta(1) + beta(2) .* x.^2);

%% Plot Data

g1 = figure;
plot(x, y, '+k'), hold on

%% Odhad parametru linearizaci modelu

beta_lin = regress(x./y, [ones(n, 1) x])';
y_lin = generator(beta_lin, x);
y_lin = filloutliers(y_lin, 'center', 'mean');

figure(g1)
plot(x, y_lin, 'xr')

%% Odhad parametru nelinearni regresi

[beta_nlin, resid, J] = nlinfit(x, y, generator, [1 1]);
y_nlin = generator(beta_nlin, x);

figure(g1)
plot(x, y_nlin, 'ob')

formatSpec = 'BETA linear: [%2.4f %2.4f], non-linear: [%2.4f %2.4f] \n';
fprintf(formatSpec, beta_lin, beta_nlin)

%% Konfidencni intervaly

% 95% intervaly spolehlivosti pro parametry beta
ci = nlparci(beta_nlin, resid, 'jacobian', J);

% linspace
xfit = linspace(min(x), max(x), 50);

% 95% predikcni interval (interval spolehlivosti)
[ypred, delta] = nlpredci(generator, xfit, beta_nlin, resid, 'jacobian', J);

% vykresleni
figure(g1)
plot(xfit, [ypred - delta, ypred + delta], 'g--')
legend('Original data', 'Linear estimation', 'Non-linear estimation', 'Confidence Intervals')
