clear; clc;

T_Real = readtable('datalogsheet.xlsx'); 
Real_Data = table2array(T_Real(:, 2:end));

T_Synth = readtable('Synthetic_PINN_Training_Data.csv');
Synth_Data = table2array(T_Synth);

figure('Name', 'RBM Generative Performance', 'Position', [100, 100, 1200, 600]);

col_idx = 2; 

subplot(1, 2, 1);
h1 = histogram(Real_Data(:, col_idx), 30, 'Normalization', 'pdf', ...
    'FaceColor', 'b', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
hold on;
h2 = histogram(Synth_Data(:, col_idx), 30, 'Normalization', 'pdf', ...
    'FaceColor', 'r', 'FaceAlpha', 0.5, 'EdgeColor', 'none');

title('Distribution Overlap: Real vs. Synthetic');
xlabel('Current (Amps)'); ylabel('Probability Density');
legend([h1, h2], {'Real Data', 'RBM Generated (10k Samples)'});
grid on;

subplot(1, 2, 2);

Corr_Real = corr(Real_Data);
Corr_Synth = corr(Synth_Data);

Difference_Map = abs(Corr_Real - Corr_Synth);

heatmap(Difference_Map, 'Colormap', jet);
title('Correlation Error Map');
xlabel('Sensor Index'); ylabel('Sensor Index');

mean_diff = mean(abs(mean(Real_Data) - mean(Synth_Data)));
fprintf('Average Mean Shift: %.4f (Lower is better)\n', mean_diff);