from glob import glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio import transforms
from tqdm import tqdm
import os

class AudioSegmentationDataset(Dataset):
    def __init__(self, source_dir, label_dir, win_sec, stride_sec, label_names, sr = 16000):
        """A Dataset class for multi-labeled audio segmentation tasks.
        Attributes:
            source_dir (str): Source directory.
            win_sec (float): Window size in sec.
            label_names (list of str): Label names.
            overlap_rate (float): Overlap rate of each neighbour windows.
            normalize (boolean): Normalize?
        """
        self.win_sec = win_sec
        self.sr = sr
        self.stride_sec = stride_sec
        # self.overlap_rate = overlap_rate
        self.label_names = label_names
        self.source_files = sorted(glob(source_dir+"/*.wav"))
        self.label_files = sorted(glob(label_dir+"/*.txt"))
        self.labels_dict = {}
        for i in range(len(label_names)):
            label = label_names[i]
            self.labels_dict[label] = i + 1 # 0 for no label
        
    def __getitem__(self, index):
        source_file = self.source_files[index]
        label_file = self.label_files[index]
        if Path(source_file).stem != Path(label_file).stem:
            raise ValueError("wav name and csv name don't match! {} {}".format(source_file, label_file))
        a, sr_from = torchaudio.load(source_file)
        a = transforms.Resample(sr_from, self.sr)(a)[0] # サンプリングレートを揃えて1チャンネル目だけを取る

        label = pd.read_table(label_file, header=None, names=["start", "end", "label", "distance"])
        label["start"] = (label["start"] * self.sr).astype(int)
        label["end"] = (label["end"] * self.sr).astype(int)
        # distanceが大きい順にソート (同じラベルが重なっている場合、近い方の距離に上書きされる)
        label = label.sort_values("distance", ascending=False)
        distance = np.zeros((len(a), len(self.label_names)))
        for _, s, e, c, d in label.itertuples():
            if c in self.label_names:
                c = self.labels_dict[c]
                distance[s:e, c-1] = d

        w = int(self.win_sec * self.sr)
        stride = int(self.stride_sec * self.sr)
        steps = ((a.shape[0] - w) // stride) - 1

        # audios = torch.as_strided(a, (steps, w), (stride,1))
        # distances = torch.as_strided(
        #     torch.tensor(distance), 
        #     (steps, w, len(self.label_names)), 
        #     (stride,1,1))

        audios = []
        distances = []
        
        for step in range(steps):
            start = step * stride
            end = start + w
            assert(a[start:end].shape[0] == w)
            distances.append(torch.tensor(distance[start:end]))
            audios.append(a[start:end])
        distances = torch.stack(distances)
        audios = torch.stack(audios)
        return audios, distances

    def __len__(self):
        return len(self.source_files)

class AudioPredictionDataset(Dataset):
    def __init__(self, source_dir, win_sec, stride_sec, sr = 16000, min_size = 100000000):
        """A Dataset class for audio prediction.
        Attributes:
            source_dir (str): Source directory.
            win_sec (float): Window size in sec.
            stride_sec (float): Window stride size in sec.
            sr (int): Sampling rate.
            min_size (int): Minimum file size.
        """
        self.win_sec = win_sec
        self.sr = sr
        self.stride_sec = stride_sec
        self.source_files = sorted(glob(source_dir+"/**/*.wav"))
        if len(self.source_files) == 0:
            self.source_files = sorted(glob(source_dir+"/*.wav"))
        print('Checking files...')
        for f in sorted(self.source_files):
            if not Path(f).exists():
                print("File not found: {}".format(f))
                # Remove the file from the list
                self.source_files.remove(f)
            if os.path.getsize(f) < min_size:
                print("File is too small: {}".format(f))
                # Remove the file from the list
                self.source_files.remove(f)
        print('Files checked!')
        print('Number of files: {}'.format(len(self.source_files)))

    def __getitem__(self, index):
        source_file = self.source_files[index]
        a, sr_from = torchaudio.load(source_file)
        a = transforms.Resample(sr_from, self.sr)(a)[0] # サンプリングレートを揃えて1チャンネル目だけを取る

        w = int(self.win_sec * self.sr)
        stride = int(self.stride_sec * self.sr)
        steps = ((a.shape[0] - w) // stride) - 1

        audios = []
        
        for step in range(steps):
            start = step * stride
            end = start + w
            assert(a[start:end].shape[0] == w)
            audios.append(a[start:end])
        audios = torch.stack(audios)
        return audios

    def __len__(self):
        return len(self.source_files)