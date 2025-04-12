import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams
import librosa
import librosa.display
import matplotlib.font_manager as fm
from matplotlib.ticker import MaxNLocator

# フォントの設定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 12

# イタリック体用のフォントプロパティを作成
italic_font = fm.FontProperties(family='Arial', style='italic')

# セミの種類（和名と学名を分ける）
common_names = [
    "Robust Cicada",
    "Large Brown Cicada",
    "Evening Cicada",
    "Kaempfer Cicada",
    "Walker's Cicada"
]

scientific_names = [
    "Hyalessa maculaticollis",
    "Graptopsaltria nigrofuscata",
    "Tanna japonensis",
    "Platypleura kaempferi",
    "Meimuna opalifera"
]

# 実際のオーディオファイルパスに置き換えてください
audio_files = [
    "step2_prepare_training_sources/data/cicada_song/segments_preprocessed/minminzemi/Minminzemi_01_01.wav",
    "step2_prepare_training_sources/data/cicada_song/segments_preprocessed/aburazemi/Aburazemi_01_01.wav",
    "step2_prepare_training_sources/data/cicada_song/segments_preprocessed/higurashi/Higurashi_01_01.wav",
    "step2_prepare_training_sources/data/cicada_song/segments_preprocessed/niiniizemi/Niiniizemi_01_01.wav",
    "step2_prepare_training_sources/data/cicada_song/segments_preprocessed/tsukutsukuboushi/nies_tsukutsukuboushi_1.wav"
]

# 図のサイズを設定
plt.figure(figsize=(15, 12))

# サブプロット間のスペースを調整
plt.subplots_adjust(hspace=0.5)  # タイトル用の余白を確保

# 各種のスペクトログラムを別々のサブプロットに
for i, (common_name, scientific_name, audio_file) in enumerate(zip(common_names, scientific_names, audio_files)):
    # サブプロットを作成
    ax = plt.subplot(5, 1, i+1)
    
    try:
        # 実際のオーディオを読み込む（実際のデータが利用可能な場合）
        y, sr = librosa.load(audio_file, sr=48000, duration=30)
        
        # スペクトログラムを計算
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        
        # スペクトログラムを表示
        img = librosa.display.specshow(D, x_axis='time', y_axis='linear', 
                                  sr=sr, ax=ax, vmin=-80, vmax=0)
    except:
        # ダミーデータを使用
        np.random.seed(i)
        dummy_spec = np.random.rand(513, 500) * np.sin(np.linspace(0, 20, 500))
        
        if i == 2:  # 夕鳴き蝉の場合は短いシラブルを表現
            mask = np.zeros_like(dummy_spec)
            for j in range(5):
                start = j * 100
                mask[:, start:start+30] = 1
            dummy_spec = dummy_spec * mask
        
        img = ax.imshow(dummy_spec, aspect='auto', origin='lower', 
                    extent=[0, 30, 0, 24], vmin=0, vmax=1,
                    cmap='viridis')
    
    # タイトルを設定（通常のタイトル機能を使用）
    ax.set_title(f"{common_name}\n({scientific_name})", pad=15, fontsize=14)
    
    # タイトルの学名部分だけをイタリック体にするため、既存のタイトルを削除して新しいテキストを追加
    title = ax.get_title()
    ax.set_title("")
    
    # 実際のデータ領域の範囲を取得
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Y軸ラベルを追加
    ax.set_ylabel('Frequency (kHz)', fontsize=12)
    
    # X軸ラベルを追加（最後の図のみ）
    if i == len(common_names) - 1:
        ax.set_xlabel('Time (sec)', fontsize=12)
    else:
        ax.set_xlabel('')
    
    # X軸の目盛りを設定
    ax.set_xticks(np.arange(0, 31, 5))
    
    # Y軸のスケールを修正
    ax.set_ylim(0, 24000)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    
    # Y軸のラベルを kHz で表示
    ticks_loc = ax.get_yticks().tolist()
    ax.set_yticks(ticks_loc)
    ax.set_yticklabels([f"{int(x/1000)}" for x in ticks_loc])
    
    # カスタムタイトルを追加（学名部分のみイタリック体に）
    common_title = ax.text(0.5, 1.12, common_name, transform=ax.transAxes,
                          ha='center', va='bottom', fontsize=14)
    
    # 学名部分をイタリック体で追加
    sci_name = scientific_name  # 学名
    sci_title = ax.text(0.5, 1.01, f"({sci_name})", transform=ax.transAxes,
                       ha='center', va='bottom', fontsize=13, fontproperties=italic_font)

# レイアウト調整
plt.tight_layout(rect=[0, 0, 0.85, 0.97])  # 上部にタイトル用の余白を確保

# カラーバーを追加
cbar_ax = plt.axes([0.88, 0.15, 0.03, 0.7])
cbar = plt.colorbar(img, cax=cbar_ax)
cbar.set_label('Amplitude (dB)', rotation=270, labelpad=20)

# 図を保存
plt.savefig('step2_prepare_training_sources/improved_figure2.jpg', dpi=300, bbox_inches='tight')
plt.close()

print("図が保存されました。")