from glob import glob
from tqdm import tqdm
import os
from pathlib import Path
import torchaudio

def preprocess_bg_files(sample_dir, out_dir, length=240):
    # Load the metadata
    samples = glob(f"{sample_dir}/**/*.wav", recursive=True)
    # Copy the sample audio files
    for sample in tqdm(samples):
        a, sr = torchaudio.load(sample)
        if "2022" in sample_dir:
            out_name = out_dir + "/2022_Suiri_" + Path(sample).parent.name + ".wav"
        else:
            out_name = out_dir + "/" + Path(sample).parent.name + "_" + Path(sample).name
        a = a[0][None, :sr*length]
        torchaudio.save(out_name, a, sr) 


sample_dirs = glob("step2_prepare_training_sources/data/background/source/*")
out_dir = "step2_prepare_training_sources/data/background/preprocessed"
os.makedirs(out_dir, exist_ok=True)
for sample_dir in sample_dirs:
    preprocess_bg_files(sample_dir, out_dir)