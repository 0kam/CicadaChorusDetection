# Step 2: Training Data Preparation

This directory contains scripts for preparing and preprocessing audio sources that will be used for chorus simulation.

## Overview

The training data consists of three types of audio sources:
1. **Cicada Songs**: Individual recordings of each target cicada species
2. **Background Sounds**: Ambient environmental sounds without cicada calls
3. **Other Sounds**: Non-cicada acoustic events (birds, insects, human-made environmental sounds)

## Scripts

- `preprocess_segments.py`: Processes cicada song recordings (resampling, filtering, normalization)
- `preprocess_backgrounds.py`: Prepares background sound files
- `calc_cicada_song_length.py`: Analyzes duration of cicada calls
- `plot_spectrograms.py`: Generates spectrograms of cicada calls for visualization

## Data Sources

The cicada song recordings were collected from:
- YouTube videos (see `step2_prepare_training_sources/youtube_list.md` for more detail)
- Audio supplements from a [field guide (ISBN: 978-4-416-61560-7)](https://www.seibundo-shinkosha.net/book/science/19749/)
- Original recordings using directional microphones

Other sounds include:
- Bird and insect calls
- Environmental sounds from ESC-50 dataset
- Human voices and music

## Preprocessing Steps

1. Resampling to 48 kHz
2. High-pass filtering (>500 Hz)
3. Normalization based on maximum amplitude
4. Segmentation into 5-30 second clips
