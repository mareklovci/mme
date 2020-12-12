%% KMA/MME Orthogonal Regression

% Clear Sequence
clear all, close all %#ok<CLALL>

% Wait 0.01s (sometimes, clear all does not delete everything)
pause(0.01)

%% Load Data

data = xlsread('regrese.xlsx');

x = data(:, 1)';
y = data(:, 2)';

figure
plot(x, y, 'xk'), grid on
xlabel('x'), ylabel('y'), title('Spotøeba piva');
hold on

% Output Table format text
formatSpec = '[deviation] probit: %3.4f, logit: %3.4f, LSE: %3.4f \n';

%% Rùzné typy regresních pøístupù

% Regression function
funkce = @(beta, x) beta(1) + beta(2) * x;

% MNC (LSE)
lsm = @(beta, x, y, funkce) norm(funkce(beta, x) - y, 2).^2;
beta_lsm = fminsearch(lsm, [0 0], [], x, y, funkce);

% Ortogonalni regrese
ortg = @(beta, x, y, funkce) norm(funkce(beta, x) - y, 2).^2 / (1 + beta(2)^2);
beta_ortg = fminsearch(ortg, [0 0], [], x, y, funkce);

% LAD regrese - nejmenší absolutní odchylka
lad = @(beta, x, y, funkce) norm(funkce(beta, x) - y, 1);
beta_lad = fminsearch(lad, [0 0], [], x, y, funkce);

plot(x, funkce(beta_lsm, x), '-r')
plot(x, funkce(beta_ortg, x), '-g')
plot(x, funkce(beta_lad, x), '-b')

legend('y pozorovani', 'LSM', 'Ortogonalni regrese', 'LAD')

% saveas(gcf, 'fig1', 'epsc')

hold off

%% Neparametrická jádrová regrese

n = length(x);
kerf = @(z) exp(-z.*z/2)/sqrt(2*pi); % Gaussova jadrova funkce
xall = linspace(min(x), max(x), 100);

h = linspace(0.025, 3, 100); % vyhlazovaci parametr

% vypocet jadrove regrese
for k = 1:length(h) % hledani optimalniho h z daneho rozpeti
   for i = 1:length(xall) % hledani hodnoty v kazdem bode x (rozkouskovani osy x)
       z = kerf(xall(i) - x / h(k)); % skalovani vzdalenosti mezi x a x0, zahrnuti meritka
       w = z' / (sum(z));
       yall(i) = y * w;
   end
   
   krivost(k) = sum(abs(diff(yall, 2))); % krivost neboli vyhlazenost
   yhat = interp1(xall, yall, x);
   s2(k) = norm(y - yhat)^2 / (length(x) - 1); % tesnost / pomoci minimalniho rozptylu
   
end

kriterium = 2*krivost / norm(krivost) + s2/norm(s2);

% vysledne optimalni h
hOpt = h(kriterium == min(kriterium));
% hOpt = 0.1;
% hOpt = 3;

for i = 1:length(xall)
   z = kerf((xall(i) - x) / hOpt);
   w = z'/sum(z);
   yall(i) = y * w;
end

% vykresleni grafu
figure
plot(x, y, 'xk'), hold on
plot(xall, yall, '-r')
legend('y', 'Neparametricka jadrova regrese')
xlabel('x')
ylabel('y')
