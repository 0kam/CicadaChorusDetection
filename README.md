# CicadaChorusDetection

This repository contains the source code for the paper "Simulation-based approach for enhancing automatic detection of cicada songs in challenging chorus conditions."

## Overview

Passive Acoustic Monitoring (PAM) has become an essential tool in ecological studies, but identifying individual species in complex choruses where multiple species vocalize simultaneously remains challenging. This project proposes a data augmentation method that uses acoustic simulation to train deep learning models capable of detecting cicada songs in chorus conditions with high accuracy.

## Features

- Acoustic simulation to generate thousands of pseudo-chorus recordings 
- Multi-label CNN-based classifier for detecting multiple cicada species
- Hyperparameter optimization for simulation parameters
- Application to phenology monitoring of five Japanese cicada species

## Core scripts
```
scripts/
├── chorus_generator.py # Chorus simuration
├── cnn.py # CNN model definition
└── utils # Utilities
    ├── data.py # Data loaders
    ├── utils.py # Custom nn.Module
    └── whombat_utils.py # Utilities for treating whombat data
```

## Project Structure

The project is organized into four main steps:

1. **Real Data Annotation**: Preparation and annotation of real-world field recordings
2. **Training Data Preparation**: Processing and segmentation of cicada songs and background sounds
3. **Model Training**: Simulation of chorus environments and CNN model training
4. **Phenology Monitoring**: Application of the trained model to analyze seasonal patterns

## Technologies Used

- PyRoomAcoustics for acoustic simulation
- PyTorch for deep learning models
- EfficientNet architecture for classification
- MLflow for experiment tracking
- Hydra and Optuna for hyperparameter tuning

## Requirements

Python and package versions are managed using [uv](https://github.com/astral-sh/uv)
```
"hydra-core>=1.3.2",
"hydra-optuna-sweeper>=1.2.0",
"librosa>=0.11.0",
"mlflow>=2.21.3",
"omegaconf>=2.3.0",
"pandas>=2.2.3",
"pyroomacoustics>=0.8.3",
"scikit-learn>=1.6.1",
"tensorboardx>=2.6.2.2",
"torch>=2.6.0",
"torch-audiomentations>=0.12.0",
"torchaudio>=2.6.0",
"torchvision>=0.21.0",
"tqdm>=4.67.1",
"whombat>=0.8.3",
```

## Citation
*Preparing!*