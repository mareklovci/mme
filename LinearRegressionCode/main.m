%% KMA/MME Linear regression

% Clear Sequence
clear all, close all %#ok<CLALL>

% Wait 0.01s (sometimes, clear all does not delete everything)
pause(0.01)

%% Initialize parameters

% Number of simulations
pocetSimulaci = 100;

% Create line and count number of data points n
x = 1:100; n = length(x);

% Choose real beta, sigma2 (rozptyl) & alpha
beta = [1, 5]; sigma2 = 2; alpha = 0.5;

% Regression Matrix X and Dependent variable Y (vysvetlovana promenna)
X = [ones(n, 1), x']; Y = X * beta';

% Preallocations
b = zeros(pocetSimulaci, 2);

% Lower & upper estimations
b_lower = zeros(pocetSimulaci, 2); b_upper = zeros(pocetSimulaci, 2);

% Variance estimation for each simulation
s2 = zeros(pocetSimulaci, 2);

% Number of estimations outside interval
pocetMimoBeta0 = 0; pocetMimoBeta1 = 0;

% Color for plot
barva = char(pocetSimulaci, 2);

%% Simulation

for i = 1:pocetSimulaci

	epsilon = randn(n, 1) * sqrt(sigma2);		%Rezidua s normalnim rozdelenim N(0;2)
	y = Y + epsilon;

    % Perform linear regression
	[btemp, bint] = regress(y, X);
    b(i,:) = btemp';
    
	e = y - X * b(i,:)';							% vypocet rezidui
	RSE = norm(e)^2;							% rezidualni soucet ctvercu pres euklidovskou normu 
	s2(i) = RSE/(n-2);							% odhad rezidualniho rozptylu
    
    Yhat = X*b(i,:)';
    resid2 = Y - Yhat;
    
	%Intervalove odhady pro regresni koeficienty Beta 0 a Beta 1
	b_lower(i, : ) = bint(:,1)';
	b_upper(i, : ) = bint(:,2)';

	if((beta(1) < b_lower(i,1)) || (beta(1) > b_upper(i,1)))
		barva(i, 1) = 'r';
		pocetMimoBeta0 = pocetMimoBeta0 + 1;
	else 
		barva(i, 1) = 'b';
	end

	if((beta(2) < b_lower(i,2)) || (beta(1) > b_upper(i,2)))
		barva(i, 2) = 'r';
		pocetMimoBeta1 = pocetMimoBeta1 + 1;
	else 
		barva(i, 2) = 'b';
	end

end

%% Plot

% Initialize figure
figure(1);

% Plot Beta 1

subplot(2,2,1)
line([0 pocetSimulaci], [beta(1) beta(1)], 'color', 'k')

for j=1:pocetSimulaci
	line([j j], [b_upper(j,1) b_lower(j,1)], 'color', barva(j, 1))
end

title("\beta_0 confidence intervals")
xlabel("Number of simulations"); ylabel("Intervals");

subplot(2,2,2)
bar([alpha * 100, pocetMimoBeta0 / pocetSimulaci * 100], 0.6);

set(gca, 'XTick', 1:2, 'XTickLabel', {'alfa', 'simulace'});
title('% of items outside interval'); ylabel('%');

% Plot Beta 2

subplot(2, 2, 3);
line([0 pocetSimulaci], [beta(2) beta(2)], 'color', 'k');

for j = 1:pocetSimulaci
	line([j j], [b_upper(j,2) b_lower(j,2)], 'color', barva(j, 2));
end

title("\beta_1 confidence intervals");
xlabel("Number of simulations"); ylabel("Intervals");

subplot(2,2,4)
bar([alpha * 100, pocetMimoBeta1 / pocetSimulaci * 100], 0.6);

set(gca, 'XTick', 1:2, 'XTickLabel', {'alfa', 'simulace'});
title('% of items outside interval'); ylabel('%');

saveas(gcf, 'fig1', 'epsc') % Save fig1

%% Normal distribution of Estimations

figure(2);
subplot(1,2,1); histfit(b(:,1));
title("\beta_0 estimations histogram");
xlabel("Estimations \beta_0"); ylabel("Number of occurences");

subplot(1,2,2); histfit(b(:,2));
title("\beta_1 estimations histogram");
xlabel("Estimations \beta_1"); ylabel("Number of occurences");

saveas(gcf, 'fig2', 'epsc') % Save fig2

%% Statistical tests of composite normality

% Lilliefors test
[h1, p1] = lillietest(b(:, 1));

% Jarque - Bera test
[h2, p2] = jbtest(b(:, 1));

% Chi2 test
[h3, p3] = chi2gof(b(:, 1));

% Print results
disp([h1 h2 h3]); disp([p1 p2 p3]);

%% Estimation Convergence to theoretical values for growing number of data

clear x n X Y epsilon b e RSE
N = 500; b = nan(N, 2);

for i = 1:N
    
    % Create line and count number of data points n
	x = 1:i*100; n = length(x);
    
    % Regression Matrix X and Dependent variable Y (vysvetlovana promenna)
	X = [ones(n,1), x']; Y = X*beta';
    
    % Residuals with N(0; 2)
	epsilon = randn(n, 1) * sqrt(sigma2);
	
    % Perform linear regression
    [btemp, bint] = regress(Y + epsilon, X); b(i,:) = btemp';
    
end

figure(3);
osax = (1:N)*100;

% Beta 0
subplot(1,2,1);

plot(osax, b(:,1), 'rx'); line([0 N*100], [beta(1) beta(1)]);
title("Estimation convergence \beta_0");
xlabel("No of Simulations"); ylabel("Confidence intervals");

% Beta 1
subplot(1,2,2);

plot(osax, b(:,2), 'rx'); line([0 N*100], [beta(2) beta(2)]);
title("Estimation convergence \beta_1");
xlabel("No of Simulations"); ylabel("Confidence intervals");

saveas(gcf, 'fig3', 'epsc') % Save fig3

% End of Script
