# Overcoming-Mode-Collapse-and-Gradient-Saturation-in-Generative-Machine-Learning-Models
Code repository for the research article **“Overcoming Mode Collapse and Gradient Saturation in Generative Machine Learning Models for Telemetry Scaling in Data-Scarce Distribution Grids”**.

This repository contains MATLAB implementations for comparing generative modeling approaches on substation/telemetry-style data, including Restricted Boltzmann Machine (RBM) variants and a Physics-Informed Neural Network (PINN) workflow.

## Overview

Pipeline for exploring how generative machine learning can be used to expand scarce telemetry data while preserving statistical structure. The repository includes:

- RBM-based synthetic data generation workflows
- A baseline Bernoulli-Bernoulli RBM implementation
- A Gaussian-Bernoulli RBM implementation
- PINN training and evaluation scripts
- Example empirical and synthetic datasets
- Comparison scripts for assessing generative quality

## Repository contents

Key files in this repository include:

- `GBRBM.m` — trains a Gaussian-Bernoulli RBM, saves the trained model, and generates synthetic telemetry samples.
- `Restricted_Botlz.m` — baseline RBM implementation using min-max scaling and a Bernoulli-Bernoulli style training loop.
- `RBM_Performance.m` — compares real vs. synthetic data distributions and correlation structure.
- `PerformanceComp.m` — compares performance between the baseline RBM and upgraded GB-RBM pipelines.
- `NewPINN.m` — trains a Physics-Informed Neural Network on synthetic training data.
- `Unified.m` — combined workflow that trains both RBM variants and PINN pipelines for side-by-side comparison.
- `BenchmarkTable.m` — benchmark or reporting utility.
- `datalogsheet.xlsx` — empirical telemetry dataset used by the MATLAB scripts.
- `Synthetic_PINN_Training_Data.csv` — synthetic dataset used for PINN training and evaluation.
- `actual_data_points.csv` — example actual/current values used in analysis and plotting.
- `LICENSE` — project license.

## Requirements

This project is implemented in **MATLAB**.

Recommended requirements:

- MATLAB R2021b or later
- Deep Learning Toolbox
- Statistics and Machine Learning Toolbox
- Spreadsheet support for `readtable` / Excel I/O
- Signal/data visualization support for `heatmap`, `histogram`, and plotting utilities

If you plan to run the PINN workflow, ensure your MATLAB installation supports:

- `dlnetwork`
- `dlarray`
- `adamupdate`
- `trainingProgressMonitor`
- `dlfeval`

## Data files

The scripts expect the following input files to be present in the working directory unless you update the paths:

- `datalogsheet.xlsx`
- `Synthetic_PINN_Training_Data.csv`
- `actual_data_points.csv`

## How to run

1. Clone the repository.
2. Open the repository folder in MATLAB.
3. Make sure the required data files are in the current working directory.
4. Run the script you want from the MATLAB editor or command window.

## Reproducibility notes

To reproduce results consistently:

- use the same MATLAB version and toolbox set across runs
- keep the input data files unchanged
- run scripts from the repository root so file paths resolve correctly
- verify that any required columns exist in the input spreadsheets/CSVs

Some scripts use randomly initialized weights and sampling, so outputs may vary slightly between runs unless you explicitly set MATLAB random seeds.
