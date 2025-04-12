# Add ./scripts to the system pathimport sys
import sys
sys.path.append("/home/okamoto/CicadaChorusDetection/scripts")
from utils.data import AudioPredictionDataset
from cnn import CNNClassifier
from torchaudio.transforms import Vol
import mlflow
import omegaconf
import pandas as pd
from pathlib import Path
import yaml
from glob import glob

d = "step3_model_training/mlruns/718625123847515763/7301479f59624b70a674ac538ea4851c/"

cfg = omegaconf.OmegaConf.load("step3_model_training/configs/tune_simulation_tpe.yaml")
logged_model = d + "/artifacts/best_model"

model = CNNClassifier(cfg)
model.model = mlflow.pytorch.load_model(logged_model)

with open('{}/meta.yaml'.format(d), 'r') as yml:
    experiment_name = yaml.safe_load(yml)["run_name"]

dir_path = "../TrueNAS/nies_audio_recording/2024/ChirpArray/Akitsu"

def predict_dir(dir_path):
    # Dataset
    dataset = AudioPredictionDataset(
        source_dir = dir_path,
        win_sec = model.c.dataset.win_sec,
        stride_sec = model.c.dataset.win_sec / 2,
        sr = 16000
    )

    transform = Vol(15, "db")

    site_name = Path(dir_path).name
    print("processing {} ...".format(site_name))
    # Prediction
    cfg.general.num_workers = 20
    preds = model.predict(dataset, transforms=transform)
    #y = preds.max(axis=1).values
    #y = (y > 0.5).float()
    df = pd.DataFrame({'file_name': dataset.source_files})
    for i, label in enumerate(model.c.dataset.label_names):
        df[label] = preds[:, i]
    out_path = '{}/{}.csv'.format("step4_application_phenology_monitoring", site_name)
    df.to_csv(out_path, index=False)


predict_dir(dir_path)