

close all
clear
clc


%%Načtení dat

data = xlsread("regrese.xlsx");

x = data(:, 1)';
y = data(:, 2)';

%Vykresleni vstupnich dat
plot(x,y, 'xk')
xlabel('x')
ylabel('y')
title('Spotřeba piva v l/věk za rok')
grid on
hold on


%% Různé typy regresních přístupů

%Regresní funkce, Teoretická funkce, kterou předpokládám ( přímka )

funkce = @(beta,x) beta(1) + beta(2)*x;


%metoda nejmenších čtverců

lsm = @(beta, x, y, funkce) norm(funkce(beta,x)-y, 2).^2;
beta_lsm = fminsearch(lsm, [0 0], [], x, y, funkce)


% ortogonální regrese

ortg = @(beta, x, y, funkce) norm(funkce(beta,x)-y, 2).^2 / (1+beta(2)^2);
beta_ortg = fminserach(ortg, [0,0], [], x, y, funkce)


%LAD regrese - nejmenší absolutní odchylka

lad = @(beta, x, y, funkce) norm(funkce(beta, x) - y, 1);
beta_lad = fminsearch(lad, [0, 0], [], x, y, funkce)


%%Vykreslení ziskaných regresních křivek
plot(x, funkce(beta_lsm, x), '-r')
plot(x, funkce(beta_ortg, x), '-g')
plot(x, funkce(beta_lad, x), '-b')
legend('y pozorovani', 'LSM', 'Ortogonalni', 'LAD')

%%Neparametrická jádrová regrese

n = length(x);
kerf = @(z) exp(-z.*z/2)/sqrt(2*pi);    % Gaussova jádrová funkce
xall = linspace(min(x), max(x), 100);

h = linspace(0.025, 3, 100) %vyhlazovací parametr

%výpočet jádrové regrese
for k = 1:length(h)					% hledání optimálního h z daného rozpětí
	for i = 1:length(xall)			% hledání hodnoty v každém bodě x ( rozkouskování osy x)
		z = kerf((xall(i)-x)/h(k))  % škálování vzdálenosti mezi x a x0, zahrnutí měřítka
		w = z'/(sum(z));			% váhy
		yall(i)-y * w;
	end

	krivost(k) = sum(abs(diff(yall, 2))); % krivost, neboli vylazenost
	yhat = interp1(xall,yall,x);		  % 
	s2(k) = norm(y-yhat)^2/(length(y)-1) %těsnost - pomocí minimálního rozptylu

end

kriterium = 2*krivost/norm(krivost) + s2/norm(s2);

% vyslene optimalni h
%hOpt = h(kriterium==min(kriterium))
%hOpt = 0.1
hOpt = 3



for i = length(xall)
	z = kerf((xall(i)-x)/hOpt);
	w = z'/(sum(z));
	yall(i) = y * w;
end

%Vykresleni grafu
%figure
plot(x,y,'xk')
hold on
plot(xall, yall, '-r')
legend('y', 'Neparametrická jádrová regrese')
xlabel('x')
ylabel('y')
grid on


























