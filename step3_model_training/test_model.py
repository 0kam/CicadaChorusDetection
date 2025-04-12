# Add ./scripts to the system path
import sys
sys.path.append("/home/okamoto/CicadaChorusDetection/scripts")
from utils.whombat_utils import WhombatDataset
from cnn import CNNClassifier
import pandas as pd
from omegaconf import OmegaConf
from glob import glob
from tqdm import tqdm
import sys
sys.path.append("/home/okamoto/CicadaChorusDetection/scripts")
import torch
import os
from torchaudio.transforms import Vol
import mlflow
from mlflow.models import infer_signature

cfg = OmegaConf.load('step3_model_training/configs/tune_simulation.yaml')

model = CNNClassifier(cfg)
model_paths = glob("step3_model_training/mlruns/388903047945866758/*")
# filter out the directories that do not contain the model.pth file
model_paths = [model_path for model_path in model_paths if os.path.exists(model_path + "/artifacts/best_model/data/model.pth")]

transform_whombat = Vol(15, "db")

# Prediction on real-world data
tune_dataset = WhombatDataset(
    proj_path = "/home/okamoto/CicadaChorusDetection/step1_real_annotation/annotation-project-e3a3c075-1f15-4d0d-bb37-58bd9e00b50b.json",
    label_names=cfg.dataset.label_names,
    win_sec=cfg.dataset.win_sec,
    sr=cfg.dataset.sr,
    transform = transform_whombat
)

test_dataset = WhombatDataset(
    proj_path="/home/okamoto/CicadaChorusDetection/step1_real_annotation/annotation-project-fd5433d4-2e19-44a7-abbd-41b82b4fe4ad.json",
    label_names=cfg.dataset.label_names,
    win_sec=cfg.dataset.win_sec,
    sr=cfg.dataset.sr,
    transform = transform_whombat
)

results = pd.DataFrame()
model_path = "step3_model_training/mlruns/388903047945866758/23493e7599364a7181c1525482d52f55"

for model_path in tqdm(model_paths):
    try:
        model.model = mlflow.pytorch.load_model(model_path + "/artifacts/best_model/")
    except FileNotFoundError:
        continue

    res_tune = model.test(tune_dataset)
    res_test = model.test(test_dataset)

    res_tune["model_path"] = model_path
    res_tune["dataset"] = "tune"

    res_test["model_path"] = model_path
    res_test["dataset"] = "test"

    results = results.append(res_tune, ignore_index=True)
    results = results.append(res_test, ignore_index=True)

results.to_csv("step3_model_training/simulation_tuning_results.csv", index=False)
