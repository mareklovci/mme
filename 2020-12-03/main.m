%% KMA/MME Linearization

% Clear Sequence
clear all, close all %#ok<CLALL>

% Wait 0.01s (sometimes, clear all does not delete everything)
pause(0.01)

%% Nelineární regrese

n = 100;
x = rand(n, 1) * 10;
beta = [2, 1];

%% Volba parametrù, tvorba modelu

funkce = @(b, x) x./(b(1) + b(2).*x);
y = funkce(beta, x);

g1 = figure;
plot(x, y, '.k')
xlabel('x'), ylabel('y'), title('Nelinearni funkce')
hold on

% pridani nahodne slozky
sigma = 0.01; % rozptyl
epsilon = sqrt(sigma) * randn(n, 1);

yeps = y + epsilon;

plot(x, yeps, '*r')

% vykresleni nahodne slozky
figure
histogram(epsilon)

% vykresleni linearizace
figure
plot(x, x./yeps, 'og')
title('Provedena linearizace')

%% Odhad parametru linearizaci modelu

beta_lin = regress(x./yeps, [ones(n, 1) x])';
y_lin = funkce(beta_lin, x);

figure(g1)
plot(x, y_lin, 'og')

%% Odhad parametru nelinearni regresi

beta_nlin = nlinfit(x, y, funkce, [1 1]);
y_nlin = funkce(beta_nlin, x);

figure(g1)
plot(x, y_nlin, '+b')

disp(beta)
disp(beta_lin)
disp(beta_nlin)

%% Vlastni odhad - nelinearni MNC
% pro zajimavost

% nejmensi b
ml = @(beta, x, yeps, funkce) norm(funkce(beta, x) - yeps, 2); % nejmenší ètverce pro závìreèné vyhodnoceni
beta_moje = fminsearch(ml, beta_lin, [], x, yeps, funkce);
y_moje = funkce(beta_moje, x);

figure(g1)
plot(x, y_moje, 'sy')
legend('data', 'data + nahodna velicina', 'Linearizace', 'Nelinearni regrese', 'Nelinearni MNC')

disp(beta)
disp(beta_lin)
disp(beta_nlin)
disp(beta_moje)
