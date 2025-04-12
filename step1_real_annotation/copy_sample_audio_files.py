import pandas as pd
from tqdm import tqdm
import os
from pathlib import Path
import torchaudio
from torchaudio.functional import highpass_biquad
import shutil

def copy_sample_audio_files(sample_csv_path, length=12, cutoff_freq=400):
    # Load the metadata
    samples = pd.read_csv(sample_csv_path)["fname"].tolist()
    out_dir = f"step1_real_annotation/recordings/{Path(sample_csv_path).stem}/"

    # Create a directory to store the sample audio files
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    # Copy the sample audio files
    for sample in tqdm(samples):
        a, sr = torchaudio.load(sample)
        out_name = out_dir + Path(sample).name
        a = a[0][None, :sr*length]
        a = highpass_biquad(a, sr, cutoff_freq=cutoff_freq)
        torchaudio.save(out_name, a, sr) 

# Copy the sample audio files
copy_sample_audio_files("step1_real_annotation/sample_akitsu_1.csv", 12, 0)
copy_sample_audio_files("step1_real_annotation/sample_ryokuchi_1.csv", 12, 0)

copy_sample_audio_files("step1_real_annotation/sample_akitsu_2.csv", 12, 0)
copy_sample_audio_files("step1_real_annotation/sample_ryokuchi_2.csv", 12, 0)

def copy_sample_audio_files2(sample_csv_path, length=12, cutoff_freq=400):
    # Load the metadata
    samples = pd.read_csv(sample_csv_path)["fname"].tolist()
    out_dir = f"step1_real_annotation/recordings/{Path(sample_csv_path).stem}/"

    # Create a directory to store the sample audio files
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    # Copy the sample audio files
    for sample in tqdm(samples):
        a, sr = torchaudio.load(sample)
        out_name = out_dir + Path(sample).parent.name + ".wav"
        a = a[0][None, :sr*length]
        a = highpass_biquad(a, sr, cutoff_freq=cutoff_freq)
        torchaudio.save(out_name, a, sr) 

copy_sample_audio_files2("step1_real_annotation/sample_suiri_1.csv", 12, 0)

copy_sample_audio_files2("step1_real_annotation/sample_suiri_2.csv", 12, 0)