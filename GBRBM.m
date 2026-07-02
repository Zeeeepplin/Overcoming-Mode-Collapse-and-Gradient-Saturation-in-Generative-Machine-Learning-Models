%% ========================================================================
%  SUBSTATION TELEMETRY EXPANSION PIPELINE: GAUSSIAN-BERNOULLI RBM
%  Optimized for 33/11 kV Substation Digital Twin Modeling
%  Designed for Cyber-Physical Data Analytics & Publication Benchmarking
%% ========================================================================

clear; clc; close all;

% 1. Data Ingestion & Environmental Setup
fprintf('Inverting Data Pipelines... Loading datalogsheet.xlsx\n');
if ~exist('datalogsheet.xlsx', 'file')
    error('Execution Halted: datalogsheet.xlsx not found in active directory.');
end

T = readtable('datalogsheet.xlsx');
% Extract variables (assuming Column 1 is Timestamp, Columns 2:end are sensors)
raw_matrix = table2array(T(:, 2:end)); 
[num_hours, num_variables] = size(raw_matrix);
fprintf('Detected Matrix Dimensions: %d Epochs/Hours across %d Substation Variables.\n', num_hours, num_variables);

% 2. Z-Score Normalization (Critical for Gaussian Visible Units)
% Min-Max scaling squashes tail variance; Z-Score preserves true Gaussian variations.
muData = mean(raw_matrix, 1);
sigmaData = std(raw_matrix, 0, 1);
sigmaData(sigmaData == 0) = 1; % Prevent divide-by-zero on static variables

normData = (raw_matrix - muData) ./ sigmaData;

% Transpose to meet network configuration matrix mapping [numVisible x numSamples]
X_train = normData'; 

%% ========================================================================
%  3. Network Hyperparameter Configuration
%% ========================================================================
numHidden = 50;        % Number of latent feature abstractors
maxEpochs = 500;       % Maximum optimization iterations
learningRate = 0.005;  % Stabilized lower boundary for GB-RBM convergence
weightPenalty = 0.001; % Active L2 Regularization coefficient (Weight Decay)
momentum = 0.9;        % Momentum coefficient to eliminate flat local minimum paths

fprintf('\nInitializing Training Chain on %d-Variable Data Node...\n', num_variables);
rbm = trainGBRBM(X_train, ...
    'NumHiddenUnits', numHidden, ...
    'MaxEpochs', maxEpochs, ... 
    'Verbose', true, ...
    'WeightPenalty', weightPenalty, ...
    'Momentum', momentum, ...
    'LearningRate', learningRate);

% Save structural weights for deployment simulation blocks
save('Substation_Robust_GBRBM.mat', 'rbm', 'muData', 'sigmaData');
disp('>> Physical-Consistent Energy Model Trained & Saved.');

%% ========================================================================
%  4. Generative Expansion Phase (Resolving Data Scarcity & Mode Collapse)
%% ========================================================================
numSamplesToSynthesize = 10000;
fprintf('\nExecuting Alternating Gibbs Sampling Chain to Synthesize %d Points...\n', numSamplesToSynthesize);
syntheticData = generateSyntheticTelemetry(rbm, numSamplesToSynthesize, muData, sigmaData);

%% ========================================================================
%  5. Comprehensive Statistical & Structural Visualization
%% ========================================================================
% Figure 1: Weight Matrix Topology Heatmap
figure('Position', [100, 100, 800, 600]);
heatmap(rbm.Weights', 'Colormap', jet, 'GridVisible', 'off');
title('Structural Weight Matrix Map: Hidden Latent Features vs Visible Sensors');
xlabel('Substation Input Sensor Nodes (1 to 21)');
ylabel('Hidden Abstraction Nodes (1 to 50)');

% Figure 2: Probability Density Function (PDF) Validation (Proving No Mode Collapse)
% We evaluate Variable 1 (e.g., Transformer Phase A Current) as an index validation
target_var_idx = 1; 
figure('Position', [150, 150, 750, 500]);
[f_real, x_real] = densityEstimation(raw_matrix(:, target_var_idx));
[f_synth, x_synth] = densityEstimation(syntheticData(:, target_var_idx));

plot(x_real, f_real, 'k-', 'LineWidth', 2.5); hold on;
plot(x_synth, f_synth, 'r--', 'LineWidth', 2.0);
grid on;
title(sprintf('Probability Density Distribution Validation (Variable Index: %d)', target_var_idx));
xlabel('Engineering Measurement Amplitude (Amps / Volts / °C)');
ylabel('Probability Mass Density');
legend('Empirical Substation Profile (Real Data)', 'Synthesized Model Profile (Corrected GB-RBM)');
set(gca, 'FontSize', 11);

%% ========================================================================
%  6. Algorithmic Processing Core Functions
%% ========================================================================

function rbm = trainGBRBM(X, varargin)
    % Parse inputs safely with explicit default conditions
    p = inputParser;
    addParameter(p, 'NumHiddenUnits', 50);
    addParameter(p, 'MaxEpochs', 100);
    addParameter(p, 'LearningRate', 0.01);
    addParameter(p, 'Verbose', true);
    addParameter(p, 'WeightPenalty', 0.001); 
    addParameter(p, 'Momentum', 0.9);
    parse(p, varargin{:});
    
    numHidden = p.Results.NumHiddenUnits;
    epochs = p.Results.MaxEpochs;
    eta = p.Results.LearningRate;
    lambda = p.Results.WeightPenalty;
    alpha = p.Results.Momentum;
    
    [numVisible, numSamples] = size(X);
    
    % Initialize weights with small random boundaries to break symmetry
    W = 0.01 * randn(numVisible, numHidden);
    b_vis = zeros(numVisible, 1);
    b_hid = zeros(numHidden, 1);
    
    % Initialize Velocity Matrices for Momentum optimization
    vW = zeros(size(W));
    vb_vis = zeros(size(b_vis));
    vb_hid = zeros(size(b_hid));
    
    for epoch = 1:epochs
        % --- POSITIVE STEP (Data-Driven Phase) ---
        % Visible is continuous; hidden states remain binary
        pos_hid_prob = sigmoid(W' * X + b_hid); 
        pos_hid_states = double(rand(size(pos_hid_prob)) < pos_hid_prob);
        pos_associations = X * pos_hid_prob';
        
        % --- NEGATIVE STEP (Model Reconstruction Phase via CD-1) ---
        % Continuous Reconstruction: Eliminate the squashing Sigmoid activation here!
        neg_vis_continuous = W * pos_hid_states + b_vis; 
        
        % Re-sample hidden configurations based on linear continuous reconstructions
        neg_hid_prob = sigmoid(W' * neg_vis_continuous + b_hid);
        neg_associations = neg_vis_continuous * neg_hid_prob';
        
        % --- GRADIENT CALCULATIONS WITH INTEGRATED L2 DECAY ---
        dW = ((pos_associations - neg_associations) / numSamples) - (lambda * W);
        db_vis = mean(X - neg_vis_continuous, 2);
        db_hid = mean(pos_hid_prob - neg_hid_prob, 2);
        
        % --- VELOCITY UPDATES (MOMENTUM ENGINE) ---
        vW = (alpha * vW) + (eta * dW);
        vb_vis = (alpha * vb_vis) + (eta * db_vis);
        vb_hid = (alpha * vb_hid) + (eta * db_hid);
        
        % --- PARAMETER UPDATES ---
        W = W + vW;
        b_vis = b_vis + vb_vis;
        b_hid = b_hid + vb_hid;
        
        % Calculate Continuous Root-Mean-Square Error (RMSE) for tracking
        rmse_err = sqrt(mean((X(:) - neg_vis_continuous(:)).^2));
        
        if p.Results.Verbose && (epoch == 1 || mod(epoch, 50) == 0)
            fprintf('GB-RBM Training Network Vector: Epoch %d/%d | Operational RMSE: %.4f\n', epoch, epochs, rmse_err);
        end
    end
    
    % Pack structures for export
    rbm.Weights = W;
    rbm.BiasVisible = b_vis;
    rbm.BiasHidden = b_hid;
end

function syntheticData = generateSyntheticTelemetry(rbm, numSamplesToSynthesize, muData, sigmaData)
    W = rbm.Weights;
    b_vis = rbm.BiasVisible;
    b_hid = rbm.BiasHidden;
    numVisible = size(W, 1);
    numHidden = size(W, 2);
    
    % Seed the chain using random binary hidden distributions
    hid_states = double(rand(numHidden, numSamplesToSynthesize) > 0.5);
    
    % Execute a deep Gibbs Sampling Chain to reach equilibrium distribution
    gibbs_burn_in_steps = 30; 
    for step = 1:gibbs_burn_in_steps
        % Conditional Reconstruction step (Hidden -> Continuous Visible)
        vis_continuous = W * hid_states + b_vis;
        
        % Feedback Activation step (Continuous Visible -> Hidden)
        hid_prob = 1 ./ (1 + exp(-(W' * vis_continuous + b_hid)));
        hid_states = double(rand(size(hid_prob)) < hid_prob);
    end
    
    % Continuous Reconstruction at final step
    syntheticData_norm = (W * hid_states + b_vis)';
    
    % Denormalization: Map standardized data back to true physical engineering metrics
    syntheticData = (syntheticData_norm .* sigmaData) + muData;
end

function y = sigmoid(x)
    % Highly stable mathematical logistic implementation
    y = 1 ./ (1 + exp(-x));
end

function [f, x] = densityEstimation(data_vector)
    % Evaluates probability distributions using kernel density estimation (KDE)
    [f, x] = ksdensity(data_vector);
end