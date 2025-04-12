import sys
sys.path.append("/home/okamoto/CicadaChorusDetection/scripts")
from utils.whombat_utils import WhombatDataset

tune_dataset = WhombatDataset(
                proj_path = "/home/okamoto/CicadaChorusDetection/step1_real_annotation/annotation-project-e3a3c075-1f15-4d0d-bb37-58bd9e00b50b.json",
                label_names=['aburazemi', 'higurashi', 'minminzemi', 'niiniizemi', 'tsukutsukuboushi'],
                win_sec=4,
                sr=16000,
                transform = None
            )

test_dataset = WhombatDataset(
    proj_path="/home/okamoto/CicadaChorusDetection/step1_real_annotation/annotation-project-fd5433d4-2e19-44a7-abbd-41b82b4fe4ad.json",
    label_names=['aburazemi', 'higurashi', 'minminzemi', 'niiniizemi', 'tsukutsukuboushi'],
    win_sec=4,
    sr=16000,
    transform=None
)

tune_dataset.df.to_csv("step1_real_annotation/tune_dataset.csv", index = False)
test_dataset.df.to_csv("step1_real_annotation/test_dataset.csv", index = False)