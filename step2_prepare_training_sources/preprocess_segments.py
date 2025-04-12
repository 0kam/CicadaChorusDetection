from glob import glob
import torchaudio
import os
from torchaudio.functional import highpass_biquad
from pathlib import Path
import torch
from scipy.signal import resample
from tqdm import tqdm
import math

def preprocess_segments(path, out, sr_to=48000):
    wav, sr = torchaudio.load(path)
    # Stereo to mono
    if wav.shape[0] == 2:
        wav = wav.mean(axis=0, keepdim=True)
    # Highpass filter
    out_dir = Path(out).parent
    sp = out_dir.parts[-1]
    if sp != 'other_sounds':
        wav = highpass_biquad(wav, sr, cutoff_freq=500)
    # Resumple
    wav = wav.numpy()[0]
    wav = resample(wav, math.floor(
        wav.shape[0] / sr * sr_to
    ))
    wav = torch.tensor(wav).unsqueeze(0)
    # Normalize the amplitude
    wav = wav / wav.abs().max() * 0.8
    # 16bit PCM
    wav = (wav * 32767).to(torch.int16)
    out_dir = Path(out).parent
    if out_dir.exists() == False:
        os.makedirs(out_dir)
    torchaudio.save(out, wav, sr_to)

files = glob('step2_prepare_training_sources/data/cicada_song/segments/*/*')
out_files = [f.replace('segments', 'segments_preprocessed') for f in files]
out_files = [f.replace(Path(f).suffix, '.wav') for f in out_files]

for f, out in tqdm(zip(files, out_files), total=len(files)):
    preprocess_segments(f, out, 48000)
