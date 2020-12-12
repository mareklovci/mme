%% KMA/MME Logit & Probit

% Clear Sequence
clear all, close all %#ok<CLALL>

% Wait 0.01s (sometimes, clear all does not delete everything)
pause(0.01)

%% Load Data

% ADMIT - bin�rn� p�ijat/nep�ijat
% GRE (Graduate Rocord Exam Scores) v�sledky z�v�re�n�ho hodnocen� na S�
% TOPNOTCH - Pat�� �kola mezi TOP
% GPA (Grade Point Average)

% �kol: sestavit model popisuj�c� �ance p�ijet� studenta na V� a odhadn�te
% parametry modelu pomoc�: probit, logit a metodou minimalizace kvadr�t�
% odchylek

input = importdata('data04_01.txt');

y = input.data(:, 1);
X = input.data(:, 2:end);

x1 = input.data(:, 2);
x2 = input.data(:, 3);
x3 = input.data(:, 4);

%% Plot 'Em All

figure
plot(x1, y, 'o'), grid on
xlabel('GRE'), ylabel('ADMIT');
saveas(gcf, '1', 'epsc');

figure
plot(x2, y, 'o'), grid on
xlabel('TOPNOTCH'), ylabel('ADMIT');
saveas(gcf, '2', 'epsc');

figure
plot(x3, y, 'o'), grid on
xlabel('GPA'), ylabel('ADMIT');
saveas(gcf, '3', 'epsc');

% Z pohledu na data zjist�me, �e 2. prom�nn� je nic ne��kaj�c�, tud�� by
% bylo vhodn� ji z modelu vy�adit

%% Line�rn� regresn� model

stats = regstats(y, X, 'linear');

% Testov�n� vhodnosti modelu pomoc� F-testu
% H0: V�echny beta = 0, HA: alespo� 1 je r�zn� od 0
F_statistika = stats.fstat.pval; 
disp(F_statistika)

% V�sledek: men�� ne� alpha => zam�t�m H0, tud�� model m� smysl

% Test, zda jsou jednotliv� koeficienty vhodn� do modelu, pomoc� t-testu
% H0: koeficient = 0
t_statistika = stats.tstat.pval;
disp(t_statistika)

% V�sledek: vektor p-hodnot ka�d�ho koef. beta modelu, hodnoty vy��� ne� 5%
% vedou k zam�tnut� H0 hypot�zy a zna�� statistickou insignifikanci dan�ho
% parametru. V tomto p��pad� je 3. hodnota vy��� ne� hladina alfa, tud��
% bychom ji m�li z modelu vy�adit

%% Logit & Probit Regrese X1

figure
plot(x1, y, 'o'), grid on, hold on
xlabel('GRE'), ylabel('ADMIT');
saveas(gcf, '4', 'epsc');

% pomocn� prostor pro vykreslen� spojit� funkce
xx = linspace(floor(min(x1)), ceil(max(x1)));

% Probit (glmfit p�id� jednotkov� sloupec automaticky)
[x1B_probit, x1Dev_probit, x1Stats_probit] = glmfit(x1, y, 'binomial', 'link', 'probit');

% x1B_probit: Odhad koeficient�
% x1Dev_probit: Rezisu�ln� sou�et �tverc�
% x1Stats_probit: Souhrnn� statistiky

% Plot Probit
plot(xx, glmval(x1B_probit, xx, 'probit'), '-r');
saveas(gcf, '5', 'epsc');

% Interpretace: p�i maxim�ln�m po�tu bod� v datech m�me st�le jen 50% �anci
% na p�ijet�

% Logit
[x1B_logit, x1Dev_logit, x1Stats_logit] = glmfit(x1, y, 'binomial', 'link', 'logit');

% Plot Logit
plot(xx, glmval(x1B_logit, xx, 'logit'), '-g');
saveas(gcf, '6', 'epsc');

% Klasick� metoda nejmen��ch �tverc�
[x1B_mnc, x1Dev_mnc, x1Stats_mnc] = glmfit(x1, y, 'binomial', 'link', 'identity');
plot(xx, glmval(x1B_mnc, xx, 'identity'), '-b'), hold off
saveas(gcf, '7', 'epsc');

disp(x1Dev_probit)
disp(x1Dev_logit)
disp(x1Dev_mnc)

% Interpretace: Rozd�ly odchylek jsou tak mal�, �e v tomto p��pad� z�vis�
% na n�s, kterou metodu zvol�me, jinak se vol� logit/probit

%% Logit & Probit Regrese X3

figure
plot(x3, y, 'o'), grid on, hold on
xlabel('GPA'), ylabel('ADMIT');
saveas(gcf, '8', 'epsc');

% pomocn� prostor pro vykreslen� spojit� funkce
xx = linspace(floor(min(x3)), ceil(max(x3)));

% Probit (glmfit p�id� jednotkov� sloupec automaticky)
[x3B_probit, x3Dev_probit, x3Stats_probit] = glmfit(x3, y, 'binomial', 'link', 'probit');

% Plot Probit
plot(xx, glmval(x3B_probit, xx, 'probit'), '-r')
saveas(gcf, '9', 'epsc');

% Logit
[x3B_logit, x3Dev_logit, x3Stats_logit] = glmfit(x3, y, 'binomial', 'link', 'logit');

% Plot Logit
plot(xx, glmval(x3B_logit, xx, 'logit'), '-g')
saveas(gcf, '10', 'epsc');

% Klasick� metoda nejmen��ch �tverc�
[x3B_mnc, x3Dev_mnc, x3Stats_mnc] = glmfit(x3, y, 'binomial', 'link', 'identity');
plot(xx, glmval(x3B_mnc, xx, 'identity'), '-b'), hold off
saveas(gcf, '11', 'epsc');

disp(x3Dev_probit)
disp(x3Dev_logit)
disp(x3Dev_mnc)

%% Model v z�vislosti na x1 a x3

Xnew = [x1, x3];

% Probit (glmfit p�id� jednotkov� sloupec automaticky)
[x13B_probit, x13Dev_probit, x13Stats_probit] = glmfit(Xnew, y, 'binomial', 'link', 'probit');

% Logit
[x13B_logit, x13Dev_logit, x13Stats_logit] = glmfit(Xnew, y, 'binomial', 'link', 'logit');

% Klasick� metoda nejmen��ch �tverc�
[x13B_mnc, x13Dev_mnc, x13Stats_mnc] = glmfit(Xnew, y, 'binomial', 'link', 'identity');

disp(x13Dev_probit)
disp(x13Dev_logit)
disp(x13Dev_mnc)