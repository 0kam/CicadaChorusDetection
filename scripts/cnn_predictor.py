from utils.utils import Transpose, Unsqueeze

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.transforms import Resize

from torchaudio import transforms
from torchaudio.functional import highpass_biquad, lowpass_biquad
import numpy as np

from omegaconf import DictConfig
import pandas as pd

class Normalize(nn.Module):
    def __init__(self) -> None:
        super(Normalize, self).__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, w, h = x.shape
        x2 = x.reshape(b, n, -1)
        x2 = (x2 - x2.mean(axis = 2).reshape(b, n, 1)) / x2.std(axis = 2).reshape(b, n, 1)
        x2 = x2.reshape(b, n, w, h)
        return x2

class CNN(nn.Module):
    def __init__(self, model_name, y_dims, n_layers, h_dims, batch_norm, drop_out):
        super().__init__()
        self.model = getattr(models, model_name)(pretrained=False)
        if "resnet" in model_name:
            self.model.fc = nn.Linear(self.model.fc.in_features, h_dims)
        elif "efficientnet" in model_name:
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, h_dims)
        if n_layers == 1:
            self.fc = nn.Linear(h_dims, y_dims)
        else:
            self.fc = [nn.Linear(h_dims, h_dims)]
            for i in range(n_layers-1):
                self.fc.append(nn.Linear(h_dims, h_dims))
                self.fc.append(nn.GELU())
                if batch_norm:
                    self.fc.append(nn.BatchNorm1d(h_dims))
                self.fc.append(nn.Dropout(drop_out))
            self.fc.append(nn.Linear(h_dims, y_dims))
            self.fc = nn.Sequential(*self.fc)
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
    
    def forward(self, x):
        h = self.cnn1(x)
        h = self.model(h)
        return self.fc(h)
        #return F.sigmoid(self.fc(h))

class CNNClassifier:
    def __init__(self, cfg: DictConfig):
        """
        Constructor for CNNClassifier
        Parameters:
        -----------
        cfg: DictConfig
            Configuration file. See cnn.yaml for more details.
        """
        print("Getting pretrained model...")
        self.c = cfg
        # Setting input size of each CNN
        if self.c.model.model_name == "efficientnet_b0":
            self.input_size = (224, 224)
        elif self.c.model.model_name == "efficientnet_b1":
            self.input_size = (240, 240)
        elif self.c.model.model_name == "efficientnet_b2":
            self.input_size = (260, 260)
        elif self.c.model.model_name == "efficientnet_b3":
            self.input_size = (300, 300)
        elif self.c.model.model_name == "efficientnet_b4":
            self.input_size = (380, 380)
        elif self.c.model.model_name == "efficientnet_b5":
            self.input_size = (456, 456)
        elif self.c.model.model_name == "efficientnet_b6":
            self.input_size = (528, 528)
        elif self.c.model.model_name == "efficientnet_b7":
            self.input_size = (600, 600)
        elif self.c.model.model_name == "efficientnet_v2_s":
            self.input_size = (384, 384)
        elif self.c.model.model_name == "efficientnet_v2_m":
            self.input_size = (480, 480)
        elif self.c.model.model_name == "efficientnet_v2_l":
            self.input_size = (480, 480)
        elif "resnet" in self.c.model.model_name:
            self.input_size = (224, 224)
        else:
            raise ValueError("Invalid model name!")

        self.build_transforms()

        # Constructing models
        print("constructing models...")
        self.model = CNN(self.c.model.model_name, len(self.c.dataset.label_names), self.c.model.n_layers, \
            self.c.model.h_dims, self.c.model.batch_norm, self.c.model.drop_out)
        self.model.to(self.c.general.device)
        if self.c.model.loss == "mse":
            self.criterion = nn.MSELoss()
        elif self.c.model.loss == "bce":
            self.criterion = nn.BCELoss()
        else:
            raise ValueError("Invalid loss function! Choose from mse, bce")
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.c.model.learning_rate)

    def predict(self, waveform: np.array, transforms = None):
        """
        Make predictions on the dataset
        Parameters:
        -----------
        waveform: np.array
            Waveform to be predicted. Shape: (n_samples). 
        
        Returns:
        --------
        predictions: torch.Tensor
            Predictions of the model
        """ 
        self.model.eval()
        if waveform.dtype == np.int16:
            waveform = waveform / 32768.0
        a = torch.tensor(waveform, dtype=torch.float32)

        if transforms is not None:
            x = transforms(a)
        else:
            x = a
        
        x = x.unsqueeze(0)
        x = x.unsqueeze(0)
        x = x.reshape(-1, 1, x.shape[2])
        
        if self.c.feature.highpass_cutoff is not None:
            x = highpass_biquad(x, self.c.dataset.sr, self.c.feature.highpass_cutoff)
        if self.c.feature.lowpass_cutoff is not None:
            x = lowpass_biquad(x, self.c.dataset.sr, self.c.feature.lowpass_cutoff)
        
        x = x[:, 0, :]
        x = self.transforms(x.to(self.c.general.device))
    
        with torch.no_grad():
            y = self.model(x).detach().numpy()
            y[0][3] = y[0][3] / 10
        
        df = pd.DataFrame(y, columns=self.c.dataset.label_names)
        return df

    def build_transforms(self):
        """
        Build transforms for the model
        transforms can vary depending on the feature type.
        The following feature types are supported:
        - mfcc
        - spectrogram
        - melspectrogram
        The obtained transforms are stored in self.transforms
        """
        if self.c.feature.feature == "mfcc":
            self.transforms = nn.Sequential(
                transforms.MFCC(self.c.dataset.sr, self.c.n_mfcc+1),
                Transpose(1,2)
            ).to(self.c.general.device)
        elif self.c.feature.feature == "spectrogram":
            self.transforms = nn.Sequential(
                #Squeeze(),
                transforms.Spectrogram(n_fft=self.c.feature.n_fft, normalized=False),
                transforms.AmplitudeToDB(),
                Transpose(1,2),
                Resize(self.input_size),
                Unsqueeze(1)
                # Normalize()
            ).to(self.c.general.device)
        elif self.c.feature.feature == "melspectrogram":
            self.transforms = nn.Sequential(
                #Squeeze(),
                transforms.MelSpectrogram(sample_rate=self.c.dataset.sr, n_fft=self.c.feature.n_fft, normalized=False, n_mels=self.c.feature.n_mels),
                transforms.AmplitudeToDB(),
                Transpose(1,2),
                Resize(self.input_size),
                Unsqueeze(1)
            ).to(self.c.general.device)
        else:
            raise ValueError("Invalid feature type! Choose from mfcc, spectrogram, melspectrogram")