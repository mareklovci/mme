%% KMA/MME Orthogonal Regression

% Clear Sequence
clear all, close all %#ok<CLALL>

% Wait 0.01s (sometimes, clear all does not delete everything)
pause(0.01)

%% Generate Model

% Output Table format text
formatSpec = '[Distance from beta] LSE: %1.4f, Ortho: %1.4f, LAD: %1.4f \n';

% Linear Regression function
linear = @(beta, x) beta(1) + beta(2) * x;

% Linear function properties
beta = [2, 1]; x = linspace(0, 10); n = length(x);

% Generate y with random chi2 residuals
y = linear(beta, x); epsilon = chi2rnd(2, [1 n]); y = y + epsilon;

% Plot that
figure
plot(x, y, 'xk'), grid on
xlabel('x'), ylabel('y'), title('Linearni model s nahodnou nenormalni rezidualni slozkou');
hold on

%% Rùzné typy regresních pøístupù

% MNC (LSE)
lsm = @(beta, x, y, funkce) norm(funkce(beta, x) - y, 2).^2;
beta_lsm = fminsearch(lsm, [0 0], [], x, y, linear);

% Ortogonalni regrese
ortg = @(beta, x, y, funkce) norm(funkce(beta, x) - y, 2).^2 / (1 + beta(2)^2);
beta_ortg = fminsearch(ortg, [0 0], [], x, y, linear);

% LAD regrese - nejmenší absolutní odchylka
lad = @(beta, x, y, funkce) norm(funkce(beta, x) - y, 1);
beta_lad = fminsearch(lad, [0 0], [], x, y, linear);

plot(x, linear(beta_lsm, x), '-r')
plot(x, linear(beta_ortg, x), '-g')
plot(x, linear(beta_lad, x), '-b')

legend('y pozorovani', 'LSM', 'Ortogonalni regrese', 'LAD')

hold off

saveas(gcf, 'fig1', 'epsc')

fprintf(formatSpec, norm(beta - beta_lsm), norm(beta - beta_ortg), norm(beta - beta_lad));

%% Neparametrická jádrová regrese

% Gaussova jadrova funkce
kerf = @(z) exp(-z.*z/2)/sqrt(2*pi);

% vyhlazovaci parametr
h = linspace(0.025, 3, 100);

% Initialize zero vectors
yall = zeros(1, n); krivost = zeros(1, n); s2 = zeros(1, n);

% vypocet jadrove regrese
for k = 1:length(h) % hledani optimalniho h z daneho rozpeti
   for i = 1:length(x) % hledani hodnoty v kazdem bode x (rozkouskovani osy x)
       z = kerf(x(i) - x / h(k)); % skalovani vzdalenosti mezi x a x0, zahrnuti meritka
       w = z' / (sum(z));
       yall(i) = y * w;
   end
   
   krivost(k) = sum(abs(diff(yall, 2))); % krivost neboli vyhlazenost
   yhat = interp1(x, yall, x);
   s2(k) = norm(y - yhat)^2 / (length(x) - 1); % tesnost pomoci minimalniho rozptylu
   
end

kriterium = 2*krivost / norm(krivost) + s2/norm(s2);

% vysledne optimalni h
hOpt = h(kriterium == min(kriterium));
% hOpt = 0.1; % hOpt = 3;

for i = 1:length(x)
   z = kerf((x(i) - x) / hOpt);
   w = z'/sum(z);
   yall(i) = y * w;
end

% vykresleni grafu
figure
plot(x, y, 'xk'), hold on
plot(x, yall, '-r')
legend('y', 'Neparametricka jadrova regrese')
xlabel('x'), ylabel('y');

saveas(gcf, 'fig2', 'epsc')
