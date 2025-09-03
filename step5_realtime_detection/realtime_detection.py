# Add ./scripts to the system pathimport sys
import sys
sys.path.append("/Users/okamoto/NIES/CicadaChorusDetection/scripts")
sys.path.append("/Users/okamoto/NIES/CicadaChorusDetection/step5_realtime_detection")
from cnn_predictor import CNNClassifier
from audio_device import CustomMic
from print_prediction import BarPlot

from torchaudio.transforms import Vol
import torch
import omegaconf
import yaml
from scipy.signal import resample

from torchaudio import save

d = "step5_realtime_detection/best_run/"

cfg = omegaconf.OmegaConf.load("step3_model_training/configs/tune_simulation_tpe.yaml")
cfg.general.device = "cpu"
logged_model = d + "/best_model.pth"

model = CNNClassifier(cfg)
model.model.load_state_dict(torch.load(logged_model, map_location=torch.device(cfg.general.device)))

with open('{}/meta.yaml'.format(d), 'r') as yml:
    experiment_name = yaml.safe_load(yml)["run_name"]

mic = CustomMic(chunk_sec=cfg.dataset.win_sec, keyword = "MacBook Proのマイク")
barplot = BarPlot(labels = ["アブラゼミ", "ヒグラシ", "ミンミンゼミ", "ニイニイゼミ", "ツクツクボウシ"], title = experiment_name, val_range = (0,1))

while True:
    waveform = mic.get() / 32768.0
    resampled_waveform = resample(waveform, cfg.dataset.sr * cfg.dataset.win_sec)
    pred = model.predict(resampled_waveform, transforms=Vol(15, "db"))
    barplot.print(pred.values[0])