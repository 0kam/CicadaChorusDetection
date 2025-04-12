# step2_prepare_training_sources/data/cicada_song/segments_preprocessed/ 以下のサブディレクトリごとに、音声の長さの合計を計算する
from glob import glob
import soundfile as sf

dirs = glob('step2_prepare_training_sources/data/cicada_song/segments_preprocessed/*')

for d in dirs:
    files = glob(f'{d}/*.wav')
    length = 0
    for f in files:
        a, sr = sf.read(f)
        length += (a.shape[0] / sr)
    print(d)
    print(len(files))
    print(length)
    print(length / len(files))

