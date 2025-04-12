library(DiagrammeR)
library(magrittr)
library(DiagrammeRsvg)
library(rsvg)

setwd("~/CicadaChorusDetection/")

# 色を定義
data_color <- "#D1E5F0"       # データソース（シリンダー）の色 - 薄い青
process_color1 <- "#C7E9C0"   # 音声処理（ボックス）の色 - 薄い緑
process_color2 <- "#FDDBC7"   # シミュレーション処理の色 - 薄いピンク
process_color3 <- "#FFFFCC"   # モデル学習・テスト処理の色 - 薄い黄色
output_color <- "#F4E3AA"     # 出力（六角形）の色 - 薄い金色

diagram <- paste0("
  digraph procedure {
    graph [rankdir = TB, fontname = 'Arial', bgcolor = 'white']
    node [fontname = 'Arial', fontsize = 12, margin = 0.2]
    edge [fontname = 'Arial', fontsize = 10]
    
    # データソース - 青色のグループ
    cicada_source[
      label = 'Focal recording of cicada songs',
      shape = cylinder,
      style = filled,
      fillcolor = '", data_color, "'
    ];
    bg_source[
      label = 'Background sound sources',
      shape = cylinder,
      style = filled,
      fillcolor = '", data_color, "'
    ];
    other_source[
      label = 'Other sound sources',
      shape = cylinder,
      style = filled,
      fillcolor = '", data_color, "'
    ];
    
    # 音声処理 - 緑色のグループ
    extract[
      label = 'Extract syllables',
      shape = box,
      style = filled,
      fillcolor = '", process_color1, "'
    ];
    resample[
      label = 'Resampling to 16 kHz',
      shape = box,
      style = filled,
      fillcolor = '", process_color1, "'
    ];
    resample_bg[
      label = 'Resampling to 16 kHz',
      shape = box,
      style = filled,
      fillcolor = '", process_color1, "'
    ];
    highpass[
      label = 'High-pass filtering (500 Hz)',
      shape = box,
      style = filled,
      fillcolor = '", process_color1, "'
    ];
    normalize[
      label = 'Normalization',
      shape = box,
      style = filled,
      fillcolor = '", process_color1, "'
    ];
    
    # 中間出力 - 黄色のグループ
    cicadas[
      label = 'Cicada songs',
      shape = hexagon,
      style = filled,
      fillcolor = '", output_color, "'
    ];
    others[
      label = 'Other sounds',
      shape = hexagon,
      style = filled,
      fillcolor = '", output_color, "'
    ];
    bgs[
      label = 'Background sounds',
      shape = hexagon,
      style = filled,
      fillcolor = '", output_color, "'
    ];

    # 他のデータソース - 青色のグループ
    realdata[
      label = 'Real chorus recordings \\n (with annotation data)',
      shape = cylinder,
      style = filled,
      fillcolor = '", data_color, "'
    ];
    params[
      label = 'Chorus simulation parameters',
      shape = cylinder,
      style = filled,
      fillcolor = '", data_color, "'
    ];
    
    # シミュレーション処理 - ピンク色のグループ
    simulate[
      label = 'Chorus simulation',
      shape = box,
      style = filled,
      fillcolor = '", process_color2, "'
    ];
    
    # シミュレーション出力 - 黄色のグループ
    chorus_source[
      label = 'Simulated chorus recordings',
      shape = hexagon,
      style = filled,
      fillcolor = '", output_color, "'
    ];
    chorus_label[
      label = 'Annotation data',
      shape = hexagon,
      style = filled,
      fillcolor = '", output_color, "'
    ];
    
    # モデル学習・テスト - 水色のグループ
    training[
      label = 'Model training',
      shape = box,
      style = filled,
      fillcolor = '", process_color3, "'
    ];
    testing[
      label = 'Model testing',
      shape = box,
      style = filled,
      fillcolor = '", process_color3, "'
    ];
    
    # 最終出力 - 黄色のグループ
    classifier[
      label = 'Classifier model',
      shape = hexagon,
      style = filled,
      fillcolor = '", output_color, "'
    ];
    accuracy[
      label = 'Model accuracy',
      shape = hexagon,
      style = filled,
      fillcolor = '", output_color, "'
    ];
   
    # エッジの定義と接続
    subgraph f {
        cicada_source -> extract -> resample -> highpass -> normalize -> cicadas
    }
    subgraph b {
        bg_source -> resample_bg -> normalize -> bgs
    }
    subgraph o {
        other_source -> resample -> highpass -> normalize -> others
    }
    
    subgraph generation {
        cicadas -> simulate
        bgs -> simulate
        others -> simulate
        params -> simulate
        simulate -> chorus_source
        simulate -> chorus_label
    }
    subgraph training {
        chorus_source -> training
        chorus_label -> training
        training -> classifier
    }
    subgraph testing {
        realdata -> testing
        classifier -> testing
        testing -> accuracy
    }
    
    # レイアウト調整
    {rank = same; cicada_source; bg_source; other_source}
    {rank = same; cicadas, bgs, others}
    {rank = same; params, simulate}
    {rank = same; realdata, testing}
    
    # 全体のグラフタイトル
    labelloc='t';
    label='Overview of the Proposed Method';
    fontsize=16;
  }
")

# SVGファイルの生成
grViz(diagram) %>%
  export_svg() %>% 
  charToRaw() %>% 
  rsvg_svg("enhanced_diagram.svg")

