# Add ./scripts to the system path
import sys
sys.path.append("/home/okamoto/CicadaChorusDetection/scripts")
from utils.utils import log_params_from_omegaconf_dict
from utils.whombat_utils import WhombatDataset
from chorus_generator import get_audio_datasets, generate
from cnn import CNNClassifier

import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
from datetime import datetime
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from torchaudio.transforms import Vol

# cfg = OmegaConf.load('/home/okamoto/CicadaChorusDetection/step3_model_training/configs/final_run.yaml')

if __name__ == "__main__":

    @hydra.main(config_path="/home/okamoto/CicadaChorusDetection/step3_model_training/configs")
    def tune_simulation(cfg: DictConfig):
        cicadas, bgs, others = get_audio_datasets(
            "/home/okamoto/CicadaChorusDetection/step2_prepare_training_sources/data/cicada_song/train",
            "/home/okamoto/CicadaChorusDetection/step2_prepare_training_sources/data/background/preprocessed",
            "/home/okamoto/CicadaChorusDetection/step2_prepare_training_sources/data/others"
        )
        # MLFlow settings
        mlflow.set_tracking_uri("file:/home/okamoto/CicadaChorusDetection/step3_model_training/mlruns")
        mlflow.set_experiment(cfg.mlflow.experiment_name)
        with mlflow.start_run(run_name="{}".format(datetime.now().strftime("%Y%m%d%H%M%S"))):
            mlflow.log_artifact(".hydra/hydra.yaml")
            mlflow.log_artifact(".hydra/overrides.yaml")
            # Data generation
            category_ratio = [cfg.generation.cicada_ratio, 1 - cfg.generation.cicada_ratio]
            
            # Data augmentation settings
            if cfg.generation.augs.pitch_shift is not None:
                pitch_shift = list(cfg.generation.augs.pitch_shift.values())
                dataaug_device = "cuda"
            else:
                pitch_shift = None
                dataaug_device = "cpu"

            def generate_chorus(wav_path, label_path):           
                generate(
                    wav_path, label_path,
                    cicadas, bgs, others,
                    48000, # Use highest sampling rate
                    cfg.generation.length,
                    category_ratio,
                    cfg.dataset.label_names,
                    cfg.generation.cicadas.weights,
                    list(cfg.generation.cicadas.popsize.values()),
                    list(cfg.generation.cicadas.distance.values()),
                    list(cfg.generation.cicadas.n_species.values()),
                    cfg.generation.others.weights,
                    list(cfg.generation.others.popsize.values()),
                    list(cfg.generation.others.distance.values()),
                    list(cfg.generation.others.n_species.values()),
                    pitch_shift_range=pitch_shift,
                    device=dataaug_device
                )
            
            # Generate train data
            train_wav_dir = cfg.generation.train_dir + '/source'
            train_label_dir = cfg.generation.train_dir + '/label'
            test_wav_dir = cfg.generation.test_dir + '/source'
            test_label_dir = cfg.generation.test_dir + '/label'

            if os.path.exists(train_wav_dir):
                os.system("rm -rf {}".format(train_wav_dir))
            if os.path.exists(train_label_dir):
                os.system("rm -rf {}".format(train_label_dir))
            
            if dataaug_device == "cpu":
                Parallel(n_jobs=4, verbose=10, require="sharedmem")([delayed(generate_chorus)(f"{train_wav_dir}/{i}.wav", f"{train_label_dir}/{i}.txt") for i in range(cfg.generation.n_train)])
            else:
                for i in tqdm(range(cfg.generation.n_train)):
                    generate_chorus(f"{train_wav_dir}/{i}.wav", f"{train_label_dir}/{i}.txt")

            # Generate test data
            if os.path.exists(test_wav_dir):
                os.system("rm -rf {}".format(test_wav_dir))
            if os.path.exists(test_label_dir):
                os.system("rm -rf {}".format(test_label_dir))
            
            if dataaug_device == "cpu":
                Parallel(n_jobs=4, verbose=10, require="sharedmem")([delayed(generate_chorus)(f"{test_wav_dir}/{i}.wav", f"{test_label_dir}/{i}.txt") for i in range(cfg.generation.n_test)])
            else:
                for i in tqdm(range(cfg.generation.n_test)):
                    generate_chorus(f"{test_wav_dir}/{i}.wav", f"{test_label_dir}/{i}.txt")
            
            # Model training
            best_val_f1_mean = 0
            model = CNNClassifier(cfg)
            for epoch in range(cfg.general.epochs):
                log_params_from_omegaconf_dict(cfg)
                train_loss = model.train()
                val_loss, res = model.val()
                mlflow.log_metric('train_loss', train_loss, step=epoch)
                mlflow.log_metric('val_loss', val_loss, step=epoch)
                f1s = []
                for c in model.c.dataset.label_names:
                    for m in ["precision", "recall", "f1"]:
                        value = res[res["label"]==c][m].values[0]
                        mlflow.log_metric('val_{}_{}'.format(m, c), value, step=epoch)
                        if m == "f1":
                            f1s.append(value)
                f1_mean = sum(f1s) / len(f1s)
                # mlflow.pytorch.log_model(model.model, "model_epoch_{}".format(epoch))
                if f1_mean >= best_val_f1_mean:
                    best_val_f1_mean = f1_mean
                    best_model = model.model
                    mlflow.pytorch.log_model(best_model, "best_model")
                    mlflow.log_metric('best_val_f1_macro', best_val_f1_mean, step=epoch)

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
                transform=transform_whombat
            )

            # Using best weights to validate model on real-world data
            model.model = best_model
            res_tune = model.test(tune_dataset)
            f1s = []
            for c in model.c.dataset.label_names:
                for m in ["precision", "recall", "f1"]:
                    value = res_tune[res_tune["label"]==c][m].values[0]
                    mlflow.log_metric('tune_{}_{}'.format(m, c), value)
                    if (m == "f1"):
                        f1s.append(value)
            tune_f1_macro = sum(f1s) / len(f1s)
            mlflow.log_metric('tune_f1_macro', tune_f1_macro)

            res_test = model.test(test_dataset)
            f1s = []
            for c in model.c.dataset.label_names:
                for m in ["precision", "recall", "f1"]:
                    value = res_test[res_test["label"]==c][m].values[0]
                    mlflow.log_metric('test_{}_{}'.format(m, c), value)
                    if (m == "f1"):
                        f1s.append(value)
            test_f1_macro = sum(f1s) / len(f1s)
            mlflow.log_metric('test_f1_macro', test_f1_macro)
        
        
        return tune_f1_macro # Use tune dataset's f1 score as the optimization metric
    
    tune_simulation()