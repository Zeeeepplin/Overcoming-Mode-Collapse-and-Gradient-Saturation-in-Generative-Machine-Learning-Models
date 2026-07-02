clear; clc;

T = readtable('datalogsheet.xlsx');
num_hours = 511;
raw_matrix = table2array(T(:, 2:end));

minVal = min(raw_matrix);
maxVal = max(raw_matrix);

denom = maxVal - minVal;
denom(denom == 0) = 1; 

normData = (raw_matrix - minVal) ./ denom;

numHidden = 50; 

fprintf('Training RBM on 15-variable Substation Data...\n');

rbm = trainRBM(normData', ...
    'NumHiddenUnits', numHidden, ...
    'MaxEpochs', 500, ... 
    'Verbose', true, ...
    'WeightPenalty', 0.001, ... 
    'LearningRate', 0.01);

save('Substation_Complex_RBM.mat', 'rbm', 'minVal', 'maxVal');
disp('Substation Model Trained & Saved.');


figure;
heatmap(rbm.Weights);
title('Correlation Map: Inputs (Rows) vs Hidden Features (Cols)');
xlabel('Hidden Features');
ylabel('Input Sensor Data (Lines & Transformers)');


function rbm = trainRBM(X, varargin)

    p = inputParser;
    addParameter(p, 'NumHiddenUnits', 10);
    addParameter(p, 'MaxEpochs', 100);
    addParameter(p, 'LearningRate', 0.1);
    addParameter(p, 'Verbose', false);
    addParameter(p, 'WeightPenalty', 0); 
    parse(p, varargin{:});
    
    numHidden = p.Results.NumHiddenUnits;
    epochs = p.Results.MaxEpochs;
    eta = p.Results.LearningRate;
    
    [numVisible, numSamples] = size(X);

    W = 0.1 * randn(numVisible, numHidden);
    b_vis = zeros(numVisible, 1);
    b_hid = zeros(numHidden, 1);

    for i = 1:epochs
 
        pos_hid_prob = sigmoid(W' * X + b_hid);
        pos_hid_states = double(rand(size(pos_hid_prob)) < pos_hid_prob);
        
        pos_associations = X * pos_hid_prob';

        neg_vis_prob = sigmoid(W * pos_hid_states + b_vis);
 
        neg_hid_prob = sigmoid(W' * neg_vis_prob + b_hid);
        
        neg_associations = neg_vis_prob * neg_hid_prob';

        W = W + eta * ((pos_associations - neg_associations) / numSamples);
        b_vis = b_vis + eta * mean(X - neg_vis_prob, 2);
        b_hid = b_hid + eta * mean(pos_hid_prob - neg_hid_prob, 2);

        err = sum(sum((X - neg_vis_prob).^2)) / numSamples;
        if p.Results.Verbose && mod(i, 50) == 0
            fprintf('Epoch %d/%d | Reconstruction Error: %.4f\n', i, epochs, err);
        end
    end

    rbm.Weights = W;
    rbm.BiasVisible = b_vis;
    rbm.BiasHidden = b_hid;
end

function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end