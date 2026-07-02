clear; clc; close all;

% data ingestion
n
fprintf('Step 1: Ingesting Substation Empirical Ground Truth...\n');
if ~exist('datalogsheet.xlsx', 'file')
    error('Execution Halted: datalogsheet.xlsx not found in the active directory.');
end

T_Real = readtable('datalogsheet.xlsx');
Real_Data = table2array(T_Real(:, 2:end));
[num_hours, num_vars] = size(Real_Data);
fprintf('>> Empirical Matrix Loaded: %d samples across %d sensor variables.\n', num_hours, num_vars);


%Pipeline A: Original Bernoulli-Bernoulli RBM Engine

fprintf('\nStep 2: Training Original Bernoulli-Bernoulli RBM Architecture...\n');

% Original Min-Max Normalization Scaling Matrix
minVal = min(Real_Data);
maxVal = max(Real_Data);
denom = maxVal - minVal;
denom(denom == 0) = 1; 
normData_Orig = (Real_Data - minVal) ./ denom;


rbm_orig = trainOriginalRBM(normData_Orig', 'NumHiddenUnits', 50, 'MaxEpochs', 300, 'LearningRate', 0.01);

fprintf('>> Generating Telemetry via Original Gibbs Chain...\n');
Synth_Orig_Norm = generateOriginalSamples(rbm_orig, 10000);
Original_Synth_Data = (Synth_Orig_Norm' .* denom) + minVal;


%Pipeline B: Upgraded Gaussian-Bernoulli RBM Engine

fprintf('\nStep 3: Training Upgraded Gaussian-Bernoulli RBM (GB-RBM) Engine...\n');

%Z-Score Normalization
muData = mean(Real_Data, 1);
sigmaData = std(Real_Data, 0, 1);
sigmaData(sigmaData == 0) = 1; 
normData_New = (Real_Data - muData) ./ sigmaData;

%Train Corrected System Configuration (with Active L2 Regularization & Momentum)
rbm_new = trainGBRBM(normData_New', 'NumHiddenUnits', 50, 'MaxEpochs', 500, 'LearningRate', 0.005, 'WeightPenalty', 0.001, 'Momentum', 0.9);

% Synthesize 10,000 samples using upgraded architecture
fprintf('>> Generating Telemetry via Continuous Energy Gibbs Chain...\n');
New_Synth_Data = generateContinuousSamples(rbm_new, 10000, muData, sigmaData);


%Statistical Metrology & Quantifications

fprintf('\nStep 4: Compiling Statistical Comparison Metrics...\n');
col_idx = 2;

% Mean Shift (First Moment Convergence)
mean_shift_orig = mean(abs(mean(Real_Data) - mean(Original_Synth_Data)));
mean_shift_new  = mean(abs(mean(Real_Data) - mean(New_Synth_Data)));

% Variance & Standard Deviation Tracking (Exposing Mode Collapse Recovery)
std_real = std(Real_Data(:, col_idx));
std_orig = std(Original_Synth_Data(:, col_idx));
std_new  = std(New_Synth_Data(:, col_idx));

std_ratio_orig = std_orig / std_real; 
std_ratio_new  = std_new / std_real;   

% Structural Correlation Modeling Error (Frobenius Matrix Difference Mappings)
Corr_Real = corr(Real_Data);
Corr_Orig = corr(Original_Synth_Data);
Corr_New  = corr(New_Synth_Data);

Diff_Map_Orig = abs(Corr_Real - Corr_Orig);
Diff_Map_New  = abs(Corr_Real - Corr_New);

frob_error_orig = norm(Diff_Map_Orig, 'fro');
frob_error_new  = norm(Diff_Map_New, 'fro');

mae_corr_orig = mean(Diff_Map_Orig(:));
mae_corr_new  = mean(Diff_Map_New(:));

% Performance Matrix


fprintf('                QUANTITATIVE PERFORMANCE BENCHMARK MATRIX            \n');
fprintf('%-35s | %-15s | %-15s\n', 'Metric Criteria Evaluated', 'Original Model', 'Upgraded GB-RBM');
fprintf('--------------------------------------------------------------------\n');
fprintf('%-35s | %-15.4f | %-15.4f\n', 'Global Average Mean Shift', mean_shift_orig, mean_shift_new);
fprintf('%-35s | %-15.4f | %-15.4f\n', ['Target Variable Std Dev (Idx: ' num2str(col_idx) ')'], std_orig, std_new);
fprintf('%-35s | %-15.4f | %-15.4f\n', 'Variance Recovery Ratio (Target)', std_ratio_orig, std_ratio_new);
fprintf('%-35s | %-15.4f | %-15.4f\n', 'Correlation Matrix Frobenius Error', frob_error_orig, frob_error_new);
fprintf('%-35s | %-15.4f | %-15.4f\n', 'Correlation Matrix MAE Error', mae_corr_orig, mae_corr_new);
fprintf('====================================================================\n');
fprintf('Target Empirical Substation Ground Truth Std Dev is: %.4f\n\n', std_real);


%Visualizations


figure('Name', 'Distribution Diagnostics', 'Position', [100, 100, 850, 480]);
[f_real, x_real] = ksdensity(Real_Data(:, col_idx));
[f_orig, x_orig] = ksdensity(Original_Synth_Data(:, col_idx));
[f_new,  x_new]  = ksdensity(New_Synth_Data(:, col_idx));

plot(x_real, f_real, 'k-', 'LineWidth', 3.0); hold on;
plot(x_orig, f_orig, 'r:', 'LineWidth', 2.5);
plot(x_new,  x_new,  'b--', 'LineWidth', 2.5);
grid on;
set(gca, 'FontSize', 11, 'FontName', 'Helvetica');
title(sprintf('Probability Density Profile Metrics (Sensor Node Index: %d)', col_idx), 'FontSize', 13);
xlabel('Substation Measurement Amplitude Units', 'FontSize', 12);
ylabel('Calculated Probability Mass Density', 'FontSize', 12);
legend({'Empirical Ground Truth (Substation)', 'Original Model (Collapsed Mode)', 'Upgraded Gaussian-Bernoulli RBM'}, 'Location', 'Best');
hold off;


figure('Name', 'Correlation Matrix Error Profiles', 'Position', [150, 150, 1100, 480]);
subplot(1, 2, 1);
imagesc(Diff_Map_Orig); colormap(jet); colorbar; caxis([0 0.8]);
title('Error Grid: Original Model', 'FontSize', 12);
xlabel('Sensor Array Node ID'); ylabel('Sensor Array Node ID'); axis square;

subplot(1, 2, 2);
imagesc(Diff_Map_New); colormap(jet); colorbar; caxis([0 0.8]);
title('Error Grid: Upgraded Gaussian-Bernoulli RBM', 'FontSize', 12);
xlabel('Sensor Array Node ID'); ylabel('Sensor Array Node ID'); axis square;

sgtitle('Absolute Inter-Sensor Topological Correlation Error Mapping: |\rho_{Real} - \rho_{Synthetic}|', 'FontSize', 14, 'FontWeight', 'bold');


%Algorithmic Processing Function


function rbm = trainOriginalRBM(X, varargin)
    p = inputParser;
    addParameter(p, 'NumHiddenUnits', 10);
    addParameter(p, 'MaxEpochs', 100);
    addParameter(p, 'LearningRate', 0.1);
    parse(p, varargin{:});
    numHidden = p.Results.NumHiddenUnits; epochs = p.Results.MaxEpochs; eta = p.Results.LearningRate;
    [numVisible, numSamples] = size(X);
    W = 0.1 * randn(numVisible, numHidden); b_vis = zeros(numVisible, 1); b_hid = zeros(numHidden, 1);
    for i = 1:epochs
        pos_hid_prob = 1 ./ (1 + exp(-(W' * X + b_hid)));
        pos_hid_states = double(rand(size(pos_hid_prob)) < pos_hid_prob);
        pos_associations = X * pos_hid_prob';
        neg_vis_prob = 1 ./ (1 + exp(-(W * pos_hid_states + b_vis)));
        neg_hid_prob = 1 ./ (1 + exp(-(W' * neg_vis_prob + b_hid)));
        neg_associations = neg_vis_prob * neg_hid_prob';
        W = W + eta * ((pos_associations - neg_associations) / numSamples);
        b_vis = b_vis + eta * mean(X - neg_vis_prob, 2);
        b_hid = b_hid + eta * mean(pos_hid_prob - neg_hid_prob, 2);
    end
    rbm.Weights = W; rbm.BiasVisible = b_vis; rbm.BiasHidden = b_hid;
end

function rbm = trainGBRBM(X, varargin)
    p = inputParser;
    addParameter(p, 'NumHiddenUnits', 50); addParameter(p, 'MaxEpochs', 100);
    addParameter(p, 'LearningRate', 0.01); addParameter(p, 'WeightPenalty', 0.001); 
    addParameter(p, 'Momentum', 0.9);
    parse(p, varargin{:});
    numHidden = p.Results.NumHiddenUnits; epochs = p.Results.MaxEpochs; eta = p.Results.LearningRate;
    lambda = p.Results.WeightPenalty; alpha = p.Results.Momentum;
    [numVisible, numSamples] = size(X);
    W = 0.01 * randn(numVisible, numHidden); b_vis = zeros(numVisible, 1); b_hid = zeros(numHidden, 1);
    vW = zeros(size(W)); vb_vis = zeros(size(b_vis)); vb_hid = zeros(size(b_hid));
    for epoch = 1:epochs
        pos_hid_prob = 1 ./ (1 + exp(-(W' * X + b_hid)));
        pos_hid_states = double(rand(size(pos_hid_prob)) < pos_hid_prob);
        pos_associations = X * pos_hid_prob';
        neg_vis_continuous = W * pos_hid_states + b_vis; 
        neg_hid_prob = 1 ./ (1 + exp(-(W' * neg_vis_continuous + b_hid)));
        neg_associations = neg_vis_continuous * neg_hid_prob';
        dW = ((pos_associations - neg_associations) / numSamples) - (lambda * W);
        db_vis = mean(X - neg_vis_continuous, 2); db_hid = mean(pos_hid_prob - neg_hid_prob, 2);
        vW = (alpha * vW) + (eta * dW); vb_vis = (alpha * vb_vis) + (eta * db_vis); vb_hid = (alpha * vb_hid) + (eta * db_hid);
        W = W + vW; b_vis = b_vis + vb_vis; b_hid = b_hid + vb_hid;
    end
    rbm.Weights = W; rbm.BiasVisible = b_vis; rbm.BiasHidden = b_hid;
end

function syntheticData = generateOriginalSamples(rbm, numSamples)
    W = rbm.Weights; b_vis = rbm.BiasVisible; b_hid = rbm.BiasHidden;
    hid_states = double(rand(size(W, 2), numSamples) > 0.5);
    for step = 1:20
        vis_prob = 1 ./ (1 + exp(-(W * hid_states + b_vis)));
        hid_prob = 1 ./ (1 + exp(-(W' * vis_prob + b_hid)));
        hid_states = double(rand(size(hid_prob)) < hid_prob);
    end
    syntheticData = 1 ./ (1 + exp(-(W * hid_states + b_vis)));
end

function syntheticData = generateContinuousSamples(rbm, numSamples, mu, sigma)
    W = rbm.Weights; b_vis = rbm.BiasVisible; b_hid = rbm.BiasHidden;
    hid_states = double(rand(size(W, 2), numSamples) > 0.5);
    for step = 1:30
        vis_continuous = W * hid_states + b_vis;
        hid_prob = 1 ./ (1 + exp(-(W' * vis_continuous + b_hid)));
        hid_states = double(rand(size(hid_prob)) < hid_prob);
    end
    norm_out = (W * hid_states + b_vis)';
    syntheticData = (norm_out .* sigma) + mu;
end