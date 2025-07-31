clc, clear, close all

cd('D:\001_Calibration\000. Experimental\Data\250325 Calibration\P001_X35Y0\Compensated')

data = readmatrix('transformed_coordinates.csv');

%%
close all
X = round(data(:,1),3);
Y = round(data(:,2),3);
Z = round(data(:,3),3);
fs = 25;
figure;
set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1]);
subplot(3,1,1)
hold on
grid
plot(X,'-o','LineWidth',5,'Color','k');
avg = round(mean(X),3);
yline(avg,'--','LineWidth',3)
text(23.1, avg,['mean = ', num2str(avg)],'FontSize',fs)
xlim([0, 23])
title('Compensated X','FontSize',fs)

subplot(3,1,2)
hold on
grid
plot(Y,'-o','LineWidth',5,'Color','k');
avg = round(mean(Y),3);
yline(avg,'--','LineWidth',3)
text(23.1, avg,['mean = ', num2str(avg)],'FontSize',fs)
xlim([0, 23])
title('Compensated Y','FontSize',fs)

subplot(3,1,3)
hold on
grid
plot(Z,'-o','LineWidth',5,'Color','k');
avg = round(mean(Z),3);
yline(avg,'--','LineWidth',3)
text(23.1, avg,['mean = ', num2str(avg)],'FontSize',fs)
xlim([0, 23])
title('Compensated Z','FontSize',fs)