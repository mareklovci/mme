%% KMA/MME Insert title

% Clear Sequence
clear all, close all %#ok<CLALL>

% Wait 0.01s (sometimes, clear all does not delete everything)
pause(0.01)

%% Import Data

input = importdata('data02_00b.txt');
y = input.data(:, 2);
X = input.data(:, 3:end);

%% Regression

output = regstats(y, X, 'linear');

%% Residuals

residuals = output.r;

% Let's plot something nice
figure
subplot(1, 2, 1); histogram(residuals); hold on
subplot(1, 2, 2); plot(residuals);

% Residuals with homogenous variance
residuals_standard = output.standres;

% Residuals with STUDENT variance
residuals_student = output.studres;

% Cook distance
cook_distance = output.cookd;

%% Projection matrix

X1 = [ones(size(X, 1)) X];

% Projection matrix given by X1 * inv(X1' * X1) * X1';
H = output.hatmat;

% Plot something nice #2
figure;
plot(diag(H), 'o'), hold on % TODO: plot "Stredni hodnota" to graph (leverage points analysis)
plot(mean(diag(H)), 'rx')
title('Diagonal Elements of the Projection Matrix H')

%% Leverage point detection
n = size(X, 1);
p = trace(H); % "Stopa matice", number of regression coeficients

% Find leverage points output.leverage = diag(H)
leverage1 = find(output.leverage > 2*p/n);

% Find leverage points with Cook distance
leverage2 = find(output.cookd > 1);

% Find leverage points with Welsch-Kuhn distance
leverage3 = find(output.dffits > 2*sqrt(p/n));

% Plot something nice #3
figure; plot(output.cookd, 'o');
title('Cook Distance')

%% Autocorrelation

figure; plot(output.r(1:end-1), output.r(2:end), 'o')
xlabel('e_i'); ylabel('e_(i+1)');

% Correlation coeficient with p-value
% There is no statistically significant correlation: rho = 0.2, p = 0.1
% Possible to get with: regress(output.r(1:end-1), output.r(2:end));
[rho, p_rho] = corr(output.r(1:end-1), output.r(2:end));

% Durbin-Watson test: H0 - residuals are not autocorrelated, HA - the are.
% alpha = 5% - I reject H0 and accept HA
% "Detekce autokorelace 1. radu"
dwstat = output.dwstat;

%% Dependence graphs & Heteroscedasticity

% Residuals to y
figure; plot(y, output.r, 'o');

% "2-vyberovy F-test shody dvou rozptylu"
% I need data in thirds, h = 1, therefore I reject H0 and accept HA
% H0: variances are the same, HA: variances are different
[h, p_h] = vartest2(output.r(y < quantile(y, 1/3)),...
                    output.r(y > quantile(y, 2/3)));

% Glejser Test "zavislost rezidui na vysvetlujicich promennych"
% Should be tested for several polynom
% Just fo info
tmp = regstats(abs(output.r), X(:,1), 'linear', 'tstat');
p_g = tmp.tstat.pval(2);                

% Residuals to x_1, x_2, x_3 dependence
figure;
subplot(3, 1, 1); plot(X(:, 1), output.r, 'o');
[~, p1] = vartest2(output.r(X(:, 1) < quantile(X(:, 1), 1/3)),...
                   output.r(X(:, 1) > quantile(X(:, 1), 2/3)));
               
subplot(3, 1, 2); plot(X(:, 2), output.r, 'o');
[~, p2] = vartest2(output.r(X(:, 2) < quantile(X(:, 2), 1/3)),...
                   output.r(X(:, 2) > quantile(X(:, 2), 2/3)));

subplot(3, 1, 3); plot(X(:, 3), output.r, 'o');
[~, p3] = vartest2(output.r(X(:, 3) < quantile(X(:, 3), 1/3)),...
                   output.r(X(:, 3) > quantile(X(:, 3), 2/3)));

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

%% sdfsdf

saveas(gcf, 'fig3', 'epsc') % Save fig3

% End of Script
