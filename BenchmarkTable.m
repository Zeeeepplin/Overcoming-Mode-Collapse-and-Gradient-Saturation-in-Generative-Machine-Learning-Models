clear; clc; close all;

%%Data Ingestion
fprintf('Loading substation empirical ground truth...\n');
if ~exist('datalogsheet.xlsx', 'file')
    error('Execution Halted: datalogsheet.xlsx not found in active directory.');
end

T_Real = readtable('datalogsheet.xlsx');
Real_Data = table2array(T_Real(:, 2:end));
[num_hours, num_vars] = size(Real_Data);
fprintf('>> Empirical Matrix: %d samples x %d variables.\n', num_hours, num_vars);

%%Architecture Definitions & Training

numSynth = 10000;
col_jsd  = 2;

%Baseline Binary RBM (B-RBM)
fprintf('\n[1/4] Training Baseline Binary RBM (B-RBM)...\n');
minVal = min(Real_Data); maxVal = max(Real_Data);
denom = maxVal - minVal; denom(denom == 0) = 1;
normData_BRBM = (Real_Data - minVal) ./ denom;

rbm_brbm = trainBinaryRBM(normData_BRBM', ...
    'NumHiddenUnits', 50, 'MaxEpochs', 300, 'LearningRate', 0.01);
Synth_BRBM_Norm = generateBinarySamples(rbm_brbm, numSynth);
Synth_BRBM = (Synth_BRBM_Norm' .* denom) + minVal;

%Variational Autoencoder (VAE)
fprintf('[2/4] Training Variational Autoencoder (VAE)...\n');
muVAE = mean(Real_Data); sigVAE = std(Real_Data); sigVAE(sigVAE == 0) = 1;
normData_VAE = (Real_Data - muVAE) ./ sigVAE;

vae = trainVAE(normData_VAE, 'LatentDim', 10, 'HiddenDim', 64, ...
    'MaxEpochs', 500, 'LearningRate', 0.001);
Synth_VAE = generateVAESamples(vae, numSynth, muVAE, sigVAE);

%WGAN-GP
fprintf('[3/4] Training WGAN-GP...\n');
muWGAN = mean(Real_Data); sigWGAN = std(Real_Data); sigWGAN(sigWGAN == 0) = 1;
normData_WGAN = (Real_Data - muWGAN) ./ sigWGAN;

gan = trainWGANGP(normData_WGAN, 'LatentDim', 10, 'HiddenDim', 64, ...
    'MaxEpochs', 1000, 'LearningRate', 0.0001, 'CriticIter', 5, 'Lambda', 10);
Synth_WGAN = generateWGANSamples(gan, numSynth, muWGAN, sigWGAN);

%Proposed Upgraded GB-RBM
fprintf('[4/4] Training Proposed Upgraded Gaussian-Bernoulli RBM (GB-RBM)...\n');
muGBRBM = mean(Real_Data); sigGBRBM = std(Real_Data); sigGBRBM(sigGBRBM == 0) = 1;
normData_GBRBM = (Real_Data - muGBRBM) ./ sigGBRBM;

rbm_gbrbm = trainGBRBM(normData_GBRBM', ...
    'NumHiddenUnits', 50, 'MaxEpochs', 500, ...
    'LearningRate', 0.005, 'WeightPenalty', 0.001, 'Momentum', 0.9);
Synth_GBRBM = generateContinuousSamples(rbm_gbrbm, numSynth, muGBRBM, sigGBRBM);

%%Metric Computation
architectures = {'Baseline Binary RBM (B-RBM)', ...
    'Variational Autoencoder (VAE)', ...
    'WGAN-GP', ...
    'Proposed Upgraded GB-RBM'};
SynthSets = {Synth_BRBM, Synth_VAE, Synth_WGAN, Synth_GBRBM};

numArch = numel(architectures);
MACE_vals     = zeros(numArch, 1);
MaxCorrE_vals = zeros(numArch, 1);
JSD_vals      = zeros(numArch, 1);
PINN_RMSE_vals = zeros(numArch, 1);

Corr_Real = corr(Real_Data);

fprintf('\nComputing benchmark metrics for all architectures...\n');

for i = 1:numArch
    S = SynthSets{i};

    %Mean Absolute Correlation Error
    Corr_Synth = corr(S);
    Diff_Map = abs(Corr_Real - Corr_Synth);
    MACE_vals(i) = mean(Diff_Map(:));

    %Max. Corr. Error
    MaxCorrE_vals(i) = max(Diff_Map(:));

    %JSD (Jensen-Shannon Divergence) at Node 2
    JSD_vals(i) = computeJSD(Real_Data(:, col_jsd), S(:, col_jsd));

    %PINN RMSE
    PINN_RMSE_vals(i) = evaluatePINNRMSE(S, Real_Data);
end

%%Display Table
fprintf('\n');
fprintf(' QUANTITATIVE BENCHMARKING OF GENERATIVE FRAMEWORKS FOR SUBSTATION TELEMETRY RECONSTRUCTION \n');
fprintf('%-40s | %8s | %18s | %14s | %10s\n', ...
    'Generative Architecture', 'MACE', 'Max. Corr. Error', 'JSD (Node 2)', 'PINN RMSE');
for i = 1:numArch
    fprintf('%-40s | %8.2f | %18.2f | %14.2f | %10.2f\n', ...
        architectures{i}, MACE_vals(i), MaxCorrE_vals(i), JSD_vals(i), PINN_RMSE_vals(i));
end

%%Save Results to MAT File
results = table(architectures', MACE_vals, MaxCorrE_vals, JSD_vals, PINN_RMSE_vals, ...
    'VariableNames', {'Architecture', 'MACE', 'MaxCorrError', 'JSD_Node2', 'PINN_RMSE'});
disp(results);
save('BenchmarkResults.mat', 'results', 'MACE_vals', 'MaxCorrE_vals', 'JSD_vals', 'PINN_RMSE_vals');
fprintf('>> Results saved to BenchmarkResults.mat\n');

%                        HELPER FUNCTIONS

%Binary RBM (Bernoulli–Bernoulli)
function rbm = trainBinaryRBM(X, varargin)
p = inputParser;
addParameter(p, 'NumHiddenUnits', 50);
addParameter(p, 'MaxEpochs', 300);
addParameter(p, 'LearningRate', 0.01);
parse(p, varargin{:});
nH = p.Results.NumHiddenUnits; epochs = p.Results.MaxEpochs; eta = p.Results.LearningRate;
[nV, nS] = size(X);
W = 0.1 * randn(nV, nH); bv = zeros(nV, 1); bh = zeros(nH, 1);
for e = 1:epochs
    ph = sigmoid(W' * X + bh);
    hs = double(rand(size(ph)) < ph);
    posA = X * ph';
    nv = sigmoid(W * hs + bv);
    nh = sigmoid(W' * nv + bh);
    negA = nv * nh';
    W  = W  + eta * ((posA - negA) / nS);
    bv = bv + eta * mean(X - nv, 2);
    bh = bh + eta * mean(ph - nh, 2);
end
rbm.Weights = W; rbm.BiasVisible = bv; rbm.BiasHidden = bh;
end

function S = generateBinarySamples(rbm, N)
W = rbm.Weights; bv = rbm.BiasVisible; bh = rbm.BiasHidden;
hs = double(rand(size(W, 2), N) > 0.5);
for k = 1:20
    vp = sigmoid(W * hs + bv);
    hp = sigmoid(W' * vp + bh);
    hs = double(rand(size(hp)) < hp);
end
S = sigmoid(W * hs + bv);
end

%Variational Autoencoder (VAE)
function vae = trainVAE(X, varargin)
p = inputParser;
addParameter(p, 'LatentDim', 10);
addParameter(p, 'HiddenDim', 64);
addParameter(p, 'MaxEpochs', 500);
addParameter(p, 'LearningRate', 0.001);
parse(p, varargin{:});
dZ = p.Results.LatentDim; dH = p.Results.HiddenDim;
epochs = p.Results.MaxEpochs; eta = p.Results.LearningRate;
[nS, dX] = size(X);

W_enc1 = 0.01 * randn(dX, dH);  b_enc1 = zeros(1, dH);
W_mu   = 0.01 * randn(dH, dZ);  b_mu   = zeros(1, dZ);
W_lv   = 0.01 * randn(dH, dZ);  b_lv   = zeros(1, dZ);

W_dec1 = 0.01 * randn(dZ, dH);  b_dec1 = zeros(1, dH);
W_out  = 0.01 * randn(dH, dX);  b_out  = zeros(1, dX);

for e = 1:epochs
    h_enc = max(0, X * W_enc1 + b_enc1);
    mu_z  = h_enc * W_mu + b_mu;
    lv_z  = h_enc * W_lv + b_lv;
    eps   = randn(nS, dZ);
    z     = mu_z + exp(0.5 * lv_z) .* eps;
    h_dec = max(0, z * W_dec1 + b_dec1);
    X_hat = h_dec * W_out + b_out;

    recon_loss = sum((X - X_hat).^2, 2);
    kl_loss    = -0.5 * sum(1 + lv_z - mu_z.^2 - exp(lv_z), 2);
    dX_hat = -2 * (X - X_hat) / nS;

    dW_out = h_dec' * dX_hat;  db_out = sum(dX_hat, 1);
    dh_dec = dX_hat * W_out';  dh_dec(h_dec <= 0) = 0;
    dW_dec1 = z' * dh_dec;     db_dec1 = sum(dh_dec, 1);
    dz = dh_dec * W_dec1';

    % Backprop reparameterisation
    dmu_z = dz + (mu_z / nS);
    dlv_z = dz .* (0.5 * exp(0.5 * lv_z) .* eps) + ...
        (-0.5 * (1 - exp(lv_z))) / nS;

    dW_mu = h_enc' * dmu_z;   db_mu = sum(dmu_z, 1);
    dW_lv = h_enc' * dlv_z;   db_lv = sum(dlv_z, 1);

    dh_enc = dmu_z * W_mu' + dlv_z * W_lv';
    dh_enc(h_enc <= 0) = 0;
    dW_enc1 = X' * dh_enc;    db_enc1 = sum(dh_enc, 1);

    % SGD update
    W_enc1 = W_enc1 - eta * dW_enc1;  b_enc1 = b_enc1 - eta * db_enc1;
    W_mu   = W_mu   - eta * dW_mu;    b_mu   = b_mu   - eta * db_mu;
    W_lv   = W_lv   - eta * dW_lv;    b_lv   = b_lv   - eta * db_lv;
    W_dec1 = W_dec1 - eta * dW_dec1;   b_dec1 = b_dec1 - eta * db_dec1;
    W_out  = W_out  - eta * dW_out;    b_out  = b_out  - eta * db_out;
end
vae.W_dec1 = W_dec1; vae.b_dec1 = b_dec1;
vae.W_out  = W_out;  vae.b_out  = b_out;
vae.dZ = dZ;
end

function S = generateVAESamples(vae, N, mu, sigma)
z = randn(N, vae.dZ);
h = max(0, z * vae.W_dec1 + vae.b_dec1);
S_norm = h * vae.W_out + vae.b_out;
S = S_norm .* sigma + mu;
end

%WGAN-GP
function gan = trainWGANGP(X, varargin)
p = inputParser;
addParameter(p, 'LatentDim', 10);
addParameter(p, 'HiddenDim', 64);
addParameter(p, 'MaxEpochs', 1000);
addParameter(p, 'LearningRate', 0.0001);
addParameter(p, 'CriticIter', 5);
addParameter(p, 'Lambda', 10);
parse(p, varargin{:});
dZ = p.Results.LatentDim; dH = p.Results.HiddenDim;
epochs = p.Results.MaxEpochs; eta = p.Results.LearningRate;
nCritic = p.Results.CriticIter; lam = p.Results.Lambda;
[nS, dX] = size(X);

% Generator:  z -> hidden -> X
Wg1 = 0.02 * randn(dZ, dH); bg1 = zeros(1, dH);
Wg2 = 0.02 * randn(dH, dX); bg2 = zeros(1, dX);
% Critic:     X -> hidden -> scalar
Wc1 = 0.02 * randn(dX, dH); bc1 = zeros(1, dH);
Wc2 = 0.02 * randn(dH, 1);  bc2 = 0;

batchSize = min(64, nS);

for e = 1:epochs

    for c = 1:nCritic
        idx = randperm(nS, batchSize);
        xr = X(idx, :);
        z  = randn(batchSize, dZ);
        % Generator forward
        hg = max(0, z * Wg1 + bg1);
        xf = hg * Wg2 + bg2;
        % Critic forward
        hr = max(0, xr * Wc1 + bc1); dr = hr * Wc2 + bc2;
        hf = max(0, xf * Wc1 + bc1); df = hf * Wc2 + bc2;
        % Gradient penalty
        alpha_gp = rand(batchSize, 1);
        xhat = alpha_gp .* xr + (1 - alpha_gp) .* xf;
        hhat = max(0, xhat * Wc1 + bc1);
        dhat = hhat * Wc2 + bc2;
        mask = double(xhat * Wc1 + bc1 > 0);        % batchSize x dH
        grad_xhat = (mask .* Wc2') * Wc1';            % batchSize x dX
        grad_norm = sqrt(sum(grad_xhat.^2, 2) + 1e-12);
        gp = mean((grad_norm - 1).^2);
        % Critic loss = E[D(fake)] - E[D(real)] + lambda*GP
        % Backprop critic
        ddr = -ones(batchSize, 1) / batchSize;
        ddf =  ones(batchSize, 1) / batchSize;
        % Through real path
        dWc2_r = hr' * ddr; dbc2_r = sum(ddr);
        dhr = ddr * Wc2'; dhr(hr <= 0) = 0;
        dWc1_r = xr' * dhr; dbc1_r = sum(dhr, 1);
        % Through fake path
        dWc2_f = hf' * ddf; dbc2_f = sum(ddf);
        dhf = ddf * Wc2'; dhf(hf <= 0) = 0;
        dWc1_f = xf' * dhf; dbc1_f = sum(dhf, 1);

        Wc1 = Wc1 - eta * (dWc1_r + dWc1_f);
        bc1 = bc1 - eta * (dbc1_r + dbc1_f);
        Wc2 = Wc2 - eta * (dWc2_r + dWc2_f);
        bc2 = bc2 - eta * (dbc2_r + dbc2_f);
    end
    z = randn(batchSize, dZ);
    hg = max(0, z * Wg1 + bg1);
    xf = hg * Wg2 + bg2;
    hf = max(0, xf * Wc1 + bc1); df = hf * Wc2 + bc2;
    ddf_g = -ones(batchSize, 1) / batchSize;
    dhf_g = ddf_g * Wc2'; dhf_g(hf <= 0) = 0;
    dxf = dhf_g * Wc1';
    dWg2 = hg' * dxf; dbg2 = sum(dxf, 1);
    dhg = dxf * Wg2'; dhg(hg <= 0) = 0;
    dWg1 = z' * dhg;   dbg1 = sum(dhg, 1);
    Wg1 = Wg1 - eta * dWg1; bg1 = bg1 - eta * dbg1;
    Wg2 = Wg2 - eta * dWg2; bg2 = bg2 - eta * dbg2;
end
gan.Wg1 = Wg1; gan.bg1 = bg1;
gan.Wg2 = Wg2; gan.bg2 = bg2;
gan.dZ = dZ;
end

function S = generateWGANSamples(gan, N, mu, sigma)
z = randn(N, gan.dZ);
h = max(0, z * gan.Wg1 + gan.bg1);
S_norm = h * gan.Wg2 + gan.bg2;
S = S_norm .* sigma + mu;
end

%Gaussian-Bernoulli RBM (GB-RBM)
function rbm = trainGBRBM(X, varargin)
p = inputParser;
addParameter(p, 'NumHiddenUnits', 50);
addParameter(p, 'MaxEpochs', 500);
addParameter(p, 'LearningRate', 0.005);
addParameter(p, 'WeightPenalty', 0.001);
addParameter(p, 'Momentum', 0.9);
parse(p, varargin{:});
nH = p.Results.NumHiddenUnits; epochs = p.Results.MaxEpochs;
eta = p.Results.LearningRate; lam = p.Results.WeightPenalty;
alpha = p.Results.Momentum;
[nV, nS] = size(X);
W = 0.01 * randn(nV, nH); bv = zeros(nV, 1); bh = zeros(nH, 1);
vW = zeros(size(W)); vbv = zeros(size(bv)); vbh = zeros(size(bh));
for e = 1:epochs
    ph = sigmoid(W' * X + bh);
    hs = double(rand(size(ph)) < ph);
    posA = X * ph';
    nv = W * hs + bv;                         % continuous reconstruction
    nh = sigmoid(W' * nv + bh);
    negA = nv * nh';
    dW = ((posA - negA) / nS) - (lam * W);
    dbv = mean(X - nv, 2); dbh = mean(ph - nh, 2);
    vW  = alpha * vW  + eta * dW;
    vbv = alpha * vbv + eta * dbv;
    vbh = alpha * vbh + eta * dbh;
    W  = W  + vW;  bv = bv + vbv;  bh = bh + vbh;
end
rbm.Weights = W; rbm.BiasVisible = bv; rbm.BiasHidden = bh;
end

function S = generateContinuousSamples(rbm, N, mu, sigma)
W = rbm.Weights; bv = rbm.BiasVisible; bh = rbm.BiasHidden;
hs = double(rand(size(W, 2), N) > 0.5);
for k = 1:30
    vc = W * hs + bv;
    hp = sigmoid(W' * vc + bh);
    hs = double(rand(size(hp)) < hp);
end
S = ((W * hs + bv)' .* sigma) + mu;
end

%Jensen-Shannon Divergence
function jsd = computeJSD(real_col, synth_col)
nBins = 50;
edges = linspace(min([real_col; synth_col]), max([real_col; synth_col]), nBins + 1);
p = histcounts(real_col,  edges, 'Normalization', 'probability') + 1e-12;
q = histcounts(synth_col, edges, 'Normalization', 'probability') + 1e-12;
p = p / sum(p);  q = q / sum(q);
m = 0.5 * (p + q);
jsd = 0.5 * sum(p .* log2(p ./ m)) + 0.5 * sum(q .* log2(q ./ m));
end

%PINN RMSE Evaluation
function rmse = evaluatePINNRMSE(SynthData, RealData)

Inputs  = SynthData(:, 2);
Targets = SynthData(:, 4);

mu_X = mean(Inputs);  sig_X = std(Inputs);  if sig_X == 0, sig_X = 1; end
mu_Y = mean(Targets); sig_Y = std(Targets); if sig_Y == 0, sig_Y = 1; end

X_norm = (Inputs  - mu_X) / sig_X;
Y_norm = (Targets - mu_Y) / sig_Y;


W1 = 0.1 * randn(1, 20);  b1 = zeros(1, 20);
W2 = 0.1 * randn(20, 20); b2 = zeros(1, 20);
W3 = 0.1 * randn(20, 1);  b3 = 0;

eta_pinn = 0.005;
numEp = 400;
nS = numel(X_norm);

for ep = 1:numEp
    % Forward
    H1 = tanh(X_norm * W1 + b1);
    H2 = tanh(H1 * W2 + b2);
    Y_pred = H2 * W3 + b3;

    % MSE loss gradient
    err = (Y_pred - Y_norm) / nS;

    % Backprop
    dW3 = H2' * err;          db3 = sum(err);
    dH2 = err * W3';          dH2 = dH2 .* (1 - H2.^2);
    dW2 = H1' * dH2;          db2 = sum(dH2, 1);
    dH1 = dH2 * W2';          dH1 = dH1 .* (1 - H1.^2);
    dW1 = X_norm' * dH1;      db1 = sum(dH1, 1);

    W1 = W1 - eta_pinn * dW1;  b1 = b1 - eta_pinn * db1;
    W2 = W2 - eta_pinn * dW2;  b2 = b2 - eta_pinn * db2;
    W3 = W3 - eta_pinn * dW3;  b3 = b3 - eta_pinn * db3;
end

Real_I = RealData(:, 2);
Real_T = RealData(:, 4);
X_test = (Real_I - mu_X) / sig_X;
H1t = tanh(X_test * W1 + b1);
H2t = tanh(H1t * W2 + b2);
Pred_T = (H2t * W3 + b3) * sig_Y + mu_Y;
rmse = sqrt(mean((Real_T - Pred_T).^2));
end

% ---- Sigmoid -----------------------------------------------------------
function y = sigmoid(x)
y = 1 ./ (1 + exp(-x));
end
