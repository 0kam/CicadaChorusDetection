# Step 3: Model Training

This directory contains scripts for generating simulated cicada choruses and training the CNN model.

## Overview

This step implements the core methodology of the paper:
1. Generating thousands of simulated cicada choruses using PyRoomAcoustics
2. Training a multi-label CNN classifier
3. Optimizing simulation parameters

## Scripts

- `run.py`: Main script for generating choruses and training the model
    For more details about models and chorus simulation, see python scripts in `../scripts/`.
    e.g., `scripts/chorus_generator.py` is a script for chorus simuration and `scripts/cnn.py` is a script for CNN models.
- `configs/`: Config files for each experiments

## Chorus Simulation

The chorus simulation places multiple sound sources in a virtual acoustic environment:
- Random selection of cicada species and individual calls
- Random placement of sound sources at different distances
- Addition of background and other environmental sounds
- Automatic generation of annotation data

Key parameters optimized through hyperparameter tuning include:
- Maximum number of cicada species per simulation
- Maximum number of cicada calls per simulation
- Distance ranges for sound sources
- Ratio of different sound types

## Model Architecture

- CNN backbone: EfficientNet-B4 (selected from CNN variants)
- Multi-label classification with sigmoid activation
- Mean Squared Error loss function

## Experiment Tracking

MLflow is used to track experiments with different:
- Simulation parameters
- Model architectures
- Training hyperparameters

Also, hyperparameters are tuned using [Hydra](https://hydra.cc/docs/intro/) and [Hydra Optuna Sweeper](https://hydra.cc/docs/plugins/optuna_sweeper/).