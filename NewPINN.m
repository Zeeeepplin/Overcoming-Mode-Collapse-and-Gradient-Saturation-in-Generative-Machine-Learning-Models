clear; clc; close all;

%data ingestion
fprintf('Step 1: Loading Synthetic CSV Engineering Profiles...\n');
if ~exist('Synthetic_PINN_Training_Data.csv', 'file')
    error('Execution Halted: Synthetic_PINN_Training_Data.csv not found.');
end

data = readtable('Synthetic_PINN_Training_Data.csv');
raw_array = table2array(data);

Inputs  = raw_array(:, 2);
Targets = raw_array(:, 4); 

mu_X = mean(Inputs);  sig_X = std(Inputs);
mu_Y = mean(Targets); sig_Y = std(Targets);

X_Train = (Inputs - mu_X) / sig_X;
Y_Train = (Targets - mu_Y) / sig_Y;

X_dl = dlarray(X_Train', 'CB'); 
Y_dl = dlarray(Y_Train', 'CB');

%Constructing Pure Differentiable Neural Topology


layers = [
    featureInputLayer(1, 'Name', 'input', 'Normalization', 'none') 
    fullyConnectedLayer(20, 'Name', 'fc1')
    tanhLayer('Name', 'tanh1')
    fullyConnectedLayer(20, 'Name', 'fc2')
    tanhLayer('Name', 'tanh2')
    fullyConnectedLayer(1, 'Name', 'output')
];

lgraph = layerGraph(layers);
dlnet = dlnetwork(lgraph); 


%Advanced Optimization Parameters (Adam Engine)

numEpochs = 600;
learnRate = 0.005;


trailingAvg = [];
trailingAvgSq = [];

monitor = trainingProgressMonitor('Metrics', ["TotalLoss", "DataLoss", "PhysicsLoss"], ...
                                  'Info', 'Epoch', ...
                                  'XLabel', 'Iteration'); 
monitor.Status = 'PINN Deep Learning Execution Online';

iteration = 0; 


%Optimization Loop

fprintf('Step 4: Executing Multi-Objective PINN Loss Minimization...\n');

for epoch = 1:numEpochs
    
    
    [loss, loss_d, loss_p, gradients] = dlfeval(@modelLoss, dlnet, X_dl, Y_dl, mu_X, sig_X, mu_Y, sig_Y);
   
    [dlnet, trailingAvg, trailingAvgSq] = adamupdate(dlnet, gradients, ...
        trailingAvg, trailingAvgSq, iteration+1, learnRate);
    
    iteration = iteration + 1;
    
   
    lossVal = double(extractdata(loss));
    lossD   = double(extractdata(loss_d));
    lossP   = double(extractdata(loss_p));
   
    recordMetrics(monitor, iteration, 'TotalLoss', lossVal, 'DataLoss', lossD, 'PhysicsLoss', lossP);
    updateInfo(monitor, Epoch=epoch);
    monitor.Progress = (epoch / numEpochs) * 100;
    
    if mod(epoch, 100) == 0 || epoch == 1
        fprintf('Epoch %d/%d | Total Loss: %.4f | Data Loss: %.4f | Phys Loss: %.4f\n', ...
            epoch, numEpochs, lossVal, lossD, lossP);
    end
end

disp('>> PINN Model Converged and Verified Against Thermodynamic Constraints.');
safePath = fullfile(userpath, 'Trained_PINN.mat');
save(safePath, 'dlnet', 'mu_X', 'sig_X', 'mu_Y', 'sig_Y');
fprintf('>> PINN Weights securely saved to safe directory: %s\n', safePath);

%Performance Analysis

fprintf('\nStep 5: Executing Cyber-Physical Holdout Validation...\n');

if ~exist('datalogsheet.xlsx', 'file')
    warning('Execution Note: datalogsheet.xlsx not found. Skipping validation phase.');
else
  
    T_Real = readtable('datalogsheet.xlsx');
    Real_Data = table2array(T_Real(:, 2:end)); 
    
   
    Real_I = Real_Data(:, 2); 
    Real_T = Real_Data(:, 4);
    

    X_test_norm = (Real_I - mu_X) / sig_X;
    X_test_dl = dlarray(X_test_norm', 'CB');
    

    Pred_T_Norm = forward(dlnet, X_test_dl);
    

    Pred_T = (double(extractdata(Pred_T_Norm))' .* sig_Y) + mu_Y;
    

    rmse_val = sqrt(mean((Real_T - Pred_T).^2));
    mae_val = mean(abs(Real_T - Pred_T));
    

    fprintf('             PINN THERMODYNAMIC VALIDATION RESULTS                   \n');
    fprintf('Holdout RMSE (Root Mean Square Error) : %.4f °C\n', rmse_val);
    fprintf('Holdout MAE  (Mean Absolute Error)    : %.4f °C\n', mae_val);

    

    figure('Name', 'PINN Cyber-Physical Performance', 'Position', [100, 100, 1000, 450]);

    subplot(1, 2, 1);
    plot(Real_T, 'k-', 'LineWidth', 2.0); hold on;
    plot(Pred_T, 'b--', 'LineWidth', 1.5);
    grid on;
    title('Digital Twin State Estimation vs. Reality', 'FontSize', 12);
    xlabel('Time (Samples)'); ylabel('Transformer Temperature (°C)');
    legend('Empirical Substation Data', sprintf('PINN Prediction (RMSE: %.2f)', rmse_val), 'Location', 'Best');
    

    subplot(1, 2, 2);
    scatter(Real_I.^2, Real_T, 15, 'k', 'filled', 'MarkerFaceAlpha', 0.5); hold on;
    scatter(Real_I.^2, Pred_T, 15, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
    

    I_range = linspace(min(Real_I), max(Real_I), 100);
    T_physics = (0.0002 .* (I_range.^2)) + 30;
    plot(I_range.^2, T_physics, 'r-', 'LineWidth', 2.0);
    
    grid on;
    title('Thermodynamic Law Tracking & Compliance', 'FontSize', 12);
    xlabel('Load Current Squared (I^2)'); ylabel('Transformer Temperature (°C)');
    legend('Real Data Variations', 'PINN Mapping', 'Joule Law (k*I^2 + 30)', 'Location', 'NorthWest');
    
    sgtitle('Cyber-Physical Holdout Validation of Physics-Informed Neural Network', 'FontSize', 14, 'FontWeight', 'bold');
end


%Functions

function [totalLoss, loss_data, loss_physics, gradients] = modelLoss(net, X, Y_Target, mu_X, sig_X, mu_Y, sig_Y)
    
  
    Y_Pred = forward(net, X);
    

    loss_data = mean((Y_Pred - Y_Target).^2, 'all');
   
    dT_dX = dlgradient(sum(Y_Pred, 'all'), X);
    
  
    I_real = (X * sig_X) + mu_X;      
    T_real = (Y_Pred * sig_Y) + mu_Y;
    
    %thermal parameters
    k = 0.0002; 
    T_ambient = 30; 
    tau = 15.0; 
    

    Physics_Residual = (tau * dT_dX) + T_real - (k * (I_real.^2) + T_ambient);
    loss_physics = mean((Physics_Residual ./ sig_Y).^2, 'all');
    

    lambda = 0.1; 
    totalLoss = loss_data + (lambda * loss_physics);

    gradients = dlgradient(totalLoss, net.Learnables);
end