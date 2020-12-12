%% KMA/MME Insert title

% Clear Sequence
clear all, close all %#ok<CLALL>

% Wait 0.01s (sometimes, clear all does not delete everything)
pause(0.01)

%% Load data with colinearity, standard analysis

load("x_m.mat");
load("y_m.mat");

X = [ones(size(x, 1), 1), x];
alpha = 0.05;
n = length(y);
p = size(X, 2);

[xx1, xx2] = meshgrid(x(:, 1), x(:, 2));

b = (X' * X) \ X' * y;
y_hat = X * b;
e = y - y_hat; % rezidua
s2 = (e' * e) / (n - p); % odhad rezidualniho rozptylu
var_b = s2 * inv(X' * X); % b * inv(A) odhad variancni matice koeficientu b

yy_hat = ones(n, n) * b(1) + xx1 * b(2) + xx2 * b(3); % vykresleni plochy regrese

% vykresleni rezidui v zavislosti na y
figure
plot(y, e, 'b*')
grid on
title('Graf rezidui')
ylabel('y')
xlabel('e_i')
legend('rezidua')

% vykresleni 3 rozmernych dat
figure
view(3)
plot3(x(:, 1), x(:, 2), y, 'ob')
grid on, hold on
mesh(xx1, xx2, yy_hat)
xlabel('x_1')
ylabel('x_2')
zlabel(y)
title('Fitted model')

%% Detekce multikolinearity

% jednoduchy postup
aux = corr(x);
corr_12 = aux(1, 2);

if abs(corr_12) >= 0.9
    print('silna multikolinearita');
else
    print('neni silna multikolinearita')
end

%% Hrebenova regrese

delta_max = (p - 1) * s2 / (b'*b); % maximalni hodnota vychyleni
b_rr = inv((X' * X) + (delta_max * eye(p))) * (X' * y);
e_rr = y - X*b_rr;
s2_rr = (e_rr' * e_rr) / (n-p);
y_hat_rr = X*b_rr;

% teoreticke hodnoty
beta = [90; -1.5; 3];
sigma_2 = 37^2;

%% Hrebenova regrese matlab

x = X(:, 2:3);
bb = ridge(y, x, delta_max, 0);

%% krokova regrese

figure
% stepwise(x, y) % interaktivni provedeni
b = stepwisefit(x, y); % automaticke provedeni

%% Dete
