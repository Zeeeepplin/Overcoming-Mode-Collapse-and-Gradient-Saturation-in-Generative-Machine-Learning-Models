clear; clc; close all;

%Data Ingestion

fprintf('Step 1: Ingesting Substation Empirical Ground Truth...\n');
if ~exist('datalogsheet.xlsx', 'file')
    error('Execution Halted: datalogsheet.xlsx not found in the active directory.');
end
T_Real = readtable('datalogsheet.xlsx');
Real_Data = table2array(T_Real(:, 2:end));
[num_hours, num_vars] = size(Real_Data);

col_I = 2; 
col_T = 4; 

%Pipeline A: Original Bernoulli-Bernoulli RBM Engine

fprintf('\nStep 2: Training Original Binary RBM & Synthesizing Data...\n');
minVal = min(Real_Data); maxVal = max(Real_Data);
denom = maxVal - minVal; denom(denom == 0) = 1; 
normData_Orig = (Real_Data - minVal) ./ denom;

rbm_orig = trainOriginalRBM(normData_Orig', 'NumHiddenUnits', 50, 'MaxEpochs', 200, 'LearningRate', 0.01);
Synth_Orig_Norm = generateOriginalSamples(rbm_orig, 5000);
Original_Synth_Data = (Synth_Orig_Norm' .* denom) + minVal;

%Upgraded Gaussian-Bernoulli RBM Engine

fprintf('\nStep 3: Training Upgraded GB-RBM & Synthesizing Data...\n');
muData = mean(Real_Data, 1); sigmaData = std(Real_Data, 0, 1);
sigmaData(sigmaData == 0) = 1; 
normData_New = (Real_Data - muData) ./ sigmaData;

rbm_new = trainGBRBM(normData_New', 'NumHiddenUnits', 50, 'MaxEpochs', 300, 'LearningRate', 0.005, 'WeightPenalty', 0.001, 'Momentum', 0.9);
New_Synth_Data = generateContinuousSamples(rbm_new, 5000, muData, sigmaData);


%Physics-Informed Neural Network (PINN) Multi-Training

fprintf('\nStep 4: Training Competing PINN Architectures...\n');
pinn_epochs = 300;
pinn_lr = 0.005;

% Train PINN A on Original Collapsed Data
fprintf('>> Training PINN A (Using Original RBM Data)...\n');
[net_Orig, mu_X_Orig, sig_X_Orig, mu_Y_Orig, sig_Y_Orig] = trainPINNEngine(...
    Original_Synth_Data(:, col_I), Original_Synth_Data(:, col_T), pinn_epochs, pinn_lr);

% Train PINN B on Upgraded GB-RBM Data
fprintf('>> Training PINN B (Using Upgraded GB-RBM Data)...\n');
[net_New, mu_X_New, sig_X_New, mu_Y_New, sig_Y_New] = trainPINNEngine(...
    New_Synth_Data(:, col_I), New_Synth_Data(:, col_T), pinn_epochs, pinn_lr);


%Final Holdout Validation against Real Substation Data

fprintf('\nStep 5: Executing Cyber-Physical Holdout Validation...\n');
Real_I = Real_Data(:, col_I);
Real_T = Real_Data(:, col_T);

X_test_orig = dlarray(((Real_I - mu_X_Orig) / sig_X_Orig)', 'CB');
Pred_T_Norm_Orig = forward(net_Orig, X_test_orig);
Pred_T_Orig = (double(extractdata(Pred_T_Norm_Orig))' .* sig_Y_Orig) + mu_Y_Orig;

X_test_new = dlarray(((Real_I - mu_X_New) / sig_X_New)', 'CB');
Pred_T_Norm_New = forward(net_New, X_test_new);
Pred_T_New = (double(extractdata(Pred_T_Norm_New))' .* sig_Y_New) + mu_Y_New;


rmse_orig = sqrt(mean((Real_T - Pred_T_Orig).^2));
rmse_new  = sqrt(mean((Real_T - Pred_T_New).^2));


%Visualization

fprintf('             PINN THERMODYNAMIC PREDICTION BENCHMARK                 \n');
fprintf('PINN Trained on Original RBM Data RMSE : %.4f °C\n', rmse_orig);
fprintf('PINN Trained on Upgraded GB-RBM Data RMSE: %.4f °C\n', rmse_new);

figure('Name', 'PINN Cyber-Physical Validation', 'Position', [150, 150, 900, 500]);
plot(Real_T, 'k-', 'LineWidth', 2.0); hold on;
plot(Pred_T_Orig, 'r--', 'LineWidth', 1.5);
plot(Pred_T_New, 'b-.', 'LineWidth', 2.0);
grid on;
title('Digital Twin Thermal Prediction Accuracy (Holdout Validation)', 'FontSize', 14);
xlabel('Temporal Samples (Time)', 'FontSize', 12);
ylabel('Transformer Temperature (°C)', 'FontSize', 12);
legend({'Empirical Substation Ground Truth', ...
        sprintf('PINN Prediction (Original RBM Data) - RMSE: %.2f', rmse_orig), ...
        sprintf('PINN Prediction (GB-RBM Data) - RMSE: %.2f', rmse_new)}, ...
        'Location', 'Best', 'FontSize', 11);
set(gca, 'FontSize', 11);
hold off;


%Functions (RBMs and PINN Engine)


function [net, mu_X, sig_X, mu_Y, sig_Y] = trainPINNEngine(I_raw, T_raw, numEpochs, learnRate)
    mu_X = mean(I_raw); sig_X = std(I_raw); sig_X(sig_X == 0) = 1;
    mu_Y = mean(T_raw); sig_Y = std(T_raw); sig_Y(sig_Y == 0) = 1;

    X_Train = (I_raw - mu_X) / sig_X;
    Y_Train = (T_raw - mu_Y) / sig_Y;
    X_dl = dlarray(X_Train', 'CB'); Y_dl = dlarray(Y_Train', 'CB');

    layers = [
        featureInputLayer(1, 'Name', 'input', 'Normalization', 'none')
        fullyConnectedLayer(20, 'Name', 'fc1')
        tanhLayer('Name', 'tanh1')
        fullyConnectedLayer(20, 'Name', 'fc2')
        tanhLayer('Name', 'tanh2')
        fullyConnectedLayer(1, 'Name', 'output')
    ];
    dlnet = dlnetwork(layerGraph(layers));
    trailingAvg = []; trailingAvgSq = [];

    for epoch = 1:numEpochs
        [~, ~, ~, gradients] = dlfeval(@modelLoss, dlnet, X_dl, Y_dl, mu_X, sig_X, mu_Y, sig_Y);
        [dlnet, trailingAvg, trailingAvgSq] = adamupdate(dlnet, gradients, trailingAvg, trailingAvgSq, epoch, learnRate);
    end
    net = dlnet;
end

function [totalLoss, loss_data, loss_physics, gradients] = modelLoss(net, X, Y_Target, mu_X, sig_X, mu_Y, sig_Y)
    Y_Pred = forward(net, X);
    loss_data = mean((Y_Pred - Y_Target).^2, 'all');
    I_real = (X * sig_X) + mu_X;      
    T_real = (Y_Pred * sig_Y) + mu_Y;
    k = 0.0002; 
    Physics_Residual = T_real - (k * (I_real.^2) + 30); 
    loss_physics = mean((Physics_Residual ./ sig_Y).^2, 'all');
    lambda = 0.5; 
    totalLoss = loss_data + (lambda * loss_physics);
    gradients = dlgradient(totalLoss, net.Learnables);
end


function rbm = trainOriginalRBM(X, varargin)
    p = inputParser; addParameter(p, 'NumHiddenUnits', 10); addParameter(p, 'MaxEpochs', 100); addParameter(p, 'LearningRate', 0.1); parse(p, varargin{:});
    numHidden = p.Results.NumHiddenUnits; epochs = p.Results.MaxEpochs; eta = p.Results.LearningRate;
    [numVisible, numSamples] = size(X);
    W = 0.1 * randn(numVisible, numHidden); b_vis = zeros(numVisible, 1); b_hid = zeros(numHidden, 1);
    for i = 1:epochs
        pos_hid_prob = 1 ./ (1 + exp(-(W' * X + b_hid))); pos_hid_states = double(rand(size(pos_hid_prob)) < pos_hid_prob); pos_associations = X * pos_hid_prob';
        neg_vis_prob = 1 ./ (1 + exp(-(W * pos_hid_states + b_vis))); neg_hid_prob = 1 ./ (1 + exp(-(W' * neg_vis_prob + b_hid))); neg_associations = neg_vis_prob * neg_hid_prob';
        W = W + eta * ((pos_associations - neg_associations) / numSamples); b_vis = b_vis + eta * mean(X - neg_vis_prob, 2); b_hid = b_hid + eta * mean(pos_hid_prob - neg_hid_prob, 2);
    end
    rbm.Weights = W; rbm.BiasVisible = b_vis; rbm.BiasHidden = b_hid;
end

function rbm = trainGBRBM(X, varargin)
    p = inputParser; addParameter(p, 'NumHiddenUnits', 50); addParameter(p, 'MaxEpochs', 100); addParameter(p, 'LearningRate', 0.01); addParameter(p, 'WeightPenalty', 0.001); addParameter(p, 'Momentum', 0.9); parse(p, varargin{:});
    numHidden = p.Results.NumHiddenUnits; epochs = p.Results.MaxEpochs; eta = p.Results.LearningRate; lambda = p.Results.WeightPenalty; alpha = p.Results.Momentum;
    [numVisible, numSamples] = size(X);
    W = 0.01 * randn(numVisible, numHidden); b_vis = zeros(numVisible, 1); b_hid = zeros(numHidden, 1); vW = zeros(size(W)); vb_vis = zeros(size(b_vis)); vb_hid = zeros(size(b_hid));
    for epoch = 1:epochs
        pos_hid_prob = 1 ./ (1 + exp(-(W' * X + b_hid))); pos_hid_states = double(rand(size(pos_hid_prob)) < pos_hid_prob); pos_associations = X * pos_hid_prob';
        neg_vis_continuous = W * pos_hid_states + b_vis; neg_hid_prob = 1 ./ (1 + exp(-(W' * neg_vis_continuous + b_hid))); neg_associations = neg_vis_continuous * neg_hid_prob';
        dW = ((pos_associations - neg_associations) / numSamples) - (lambda * W); db_vis = mean(X - neg_vis_continuous, 2); db_hid = mean(pos_hid_prob - neg_hid_prob, 2);
        vW = (alpha * vW) + (eta * dW); vb_vis = (alpha * vb_vis) + (eta * db_vis); vb_hid = (alpha * vb_hid) + (eta * db_hid);
        W = W + vW; b_vis = b_vis + vb_vis; b_hid = b_hid + vb_hid;
    end
    rbm.Weights = W; rbm.BiasVisible = b_vis; rbm.BiasHidden = b_hid;
end

function syntheticData = generateOriginalSamples(rbm, numSamples)
    W = rbm.Weights; b_vis = rbm.BiasVisible; b_hid = rbm.BiasHidden; hid_states = double(rand(size(W, 2), numSamples) > 0.5);
    for step = 1:20
        vis_prob = 1 ./ (1 + exp(-(W * hid_states + b_vis))); hid_prob = 1 ./ (1 + exp(-(W' * vis_prob + b_hid))); hid_states = double(rand(size(hid_prob)) < hid_prob);
    end
    syntheticData = 1 ./ (1 + exp(-(W * hid_states + b_vis)));
end

function syntheticData = generateContinuousSamples(rbm, numSamples, mu, sigma)
    W = rbm.Weights; b_vis = rbm.BiasVisible; b_hid = rbm.BiasHidden; hid_states = double(rand(size(W, 2), numSamples) > 0.5);
    for step = 1:30
        vis_continuous = W * hid_states + b_vis; hid_prob = 1 ./ (1 + exp(-(W' * vis_continuous + b_hid))); hid_states = double(rand(size(hid_prob)) < hid_prob);
    end
    norm_out = (W * hid_states + b_vis)'; syntheticData = (norm_out .* sigma) + mu;
end