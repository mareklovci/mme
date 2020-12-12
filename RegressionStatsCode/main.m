%% KMA/MME Model Quality Assesment

% Clear Sequence
clear all, close all %#ok<CLALL>

% Wait 0.01s (sometimes, clear all does not delete everything)
pause(0.01)

%% Load Data
% Price of Monet Paintings

% (price, height, width, signed, picture, house)
input = importdata('TableF4-1.csv');

y = input.data(:, 1);
X = input.data(:, 2:4);

n = size(X, 1);

%% Plot Data

figure
subplot(3, 1, 1), plot(X(:, 1), y, '.'), grid on
xlabel('Height [inches]'); ylabel('Price [mil. $]');

hold on

subplot(3, 1, 2), plot(X(:, 2), y, '.'), grid on
xlabel('Width [inches]'); ylabel('Price [mil. $]');

subplot(3, 1, 3), plot(X(:, 3), y, 'o'), grid on
xlabel('Signature (0 = No, 1 = Yes)'); ylabel('Price [mil. $]');

hold off

saveas(gcf, 'fig1', 'epsc')

%% Linear Model

[b, bint] = regress(y, [ones(n, 1), X]);

%% Regression Diagnostics

output = regstats(y, X, 'linear');

%% Leverage point detection

% Projection matrix given by X1 * inv(X1' * X1) * X1';
H = output.hatmat;
p = trace(H); % "Stopa matice", number of regression coeficients

% Find leverage points output.leverage = diag(H)
leverage1 = find(output.leverage > 2*p/n);

% Find leverage points with Cook distance
leverage2 = find(output.cookd > 4/n);

% Find leverage points with Welsch-Kuhn distance
leverage3 = find(output.dffits > 2*sqrt(p/n));

% Plot Cook Distance
figure; bar(output.cookd);
title('Cook Distance'), hold on;
ylabel('Cook distance'); xlabel('Data Point');
plot([0 n], [4/n 4/n], '--'); hold off;

saveas(gcf, 'fig2', 'epsc')

%% Residuals Normality

residuals = output.r;

% Lilliefors test
[h1, p1] = lillietest(residuals);

% Jarque - Bera test
[h2, p2] = jbtest(residuals);

% Chi2 test
[h3, p3] = chi2gof(residuals);

diary on

% Print Hypothesis and p-values
fprintf('H1 = %1.0f, P1 = %1.3f\n', [h1, p1])
fprintf('H2 = %1.0f, P2 = %1.3f\n', [h2, p2])
fprintf('H3 = %1.0f, P3 = %1.3f\n', [h3, p3])

diary off

%% Normal distribution of Residuals

figure; histfit(residuals);
title("Histogram of Residuals");
xlabel("Residual value"); ylabel("Number of occurences");

saveas(gcf, 'fig3', 'epsc')

%% Linearity Test

% Residuals to y
figure; plot(y, output.studres, '.');
title('Lineary predicted y to studentized residuals')

saveas(gcf, 'fig4', 'epsc')

%% Heteroscedasticity

% "2-vyberovy F-test shody dvou rozptylu"
% I need data in thirds, h = 1, therefore I reject H0 and accept HA
% H0: variances are the same, HA: variances are different
[h, p_h] = vartest2(output.r(y < quantile(y, 1/3)),...
                    output.r(y > quantile(y, 2/3)));

%% Dependence graphs

% Residuals to x_1, x_2, x_3 dependence
figure;
subplot(3, 1, 1); plot(X(:, 1), output.r, '.'); grid on;
[~, pr1] = vartest2(output.r(X(:, 1) < quantile(X(:, 1), 1/3)),...
                    output.r(X(:, 1) > quantile(X(:, 1), 2/3)));
               
subplot(3, 1, 2); plot(X(:, 2), output.r, '.'); grid on;
[~, pr2] = vartest2(output.r(X(:, 2) < quantile(X(:, 2), 1/3)),...
                    output.r(X(:, 2) > quantile(X(:, 2), 2/3)));

subplot(3, 1, 3); plot(X(:, 3), output.r, 'o'); grid on;
[~, pr3] = vartest2(output.r(X(:, 3) < quantile(X(:, 3), 1/3)),...
                    output.r(X(:, 3) > quantile(X(:, 3), 2/3)));

saveas(gcf, 'fig5', 'epsc')

%% Multicolinearity

% metoda parovych koeficientu korelace
[rho, pval] = corr(X); % H0: rho = 0 ... tj. statisticky nevyznamny

% metoda pomocnych regresi - zavislou promennou jsou postupne vsechny
% vysvetlujici promenne

Rj = zeros(size(X, 2), 1);

for j = 1:size(X, 2)
   Xj = [X(:, 1:j-1) X(:, j+1:size(X, 2))]; 
   R = regstats(X(:, j), Xj, 'linear', 'rsquare');
   Rj(j) = R.rsquare;
end

VIF = 1 ./ (1-Rj);

%% Autocorrelation

figure; plot(output.r(1:end-1), output.r(2:end), '.')
xlabel('e_i'); ylabel('e_(i+1)');

% End of Script