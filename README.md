# dsmaker — DiffSinger Voice Model Builder

音声ファイルと歌詞を入れるだけで、OpenUtau用のDiffSinger歌声モデルを作成できるツールです。

## インストール

```bash
pip install git+https://github.com/tererun/diffsinger-maker.git
```

Web UIも使う場合:
```bash
pip install "git+https://github.com/tererun/diffsinger-maker.git#egg=diffsinger-maker[ui]"
```

### 必要環境
- Python 3.10以上
- (学習する場合) NVIDIA GPU + CUDA、または Google Colab

## 使い方

### Web UI（おすすめ）

```bash
dsmaker ui
```

ブラウザで `http://127.0.0.1:7860` が開きます。画面の指示に従って操作するだけです。

### ワンライナー（CLI）

```bash
dsmaker make --audio ./songs/ --lyrics ./lyrics/ --name "MySinger" --lang ja
```

### 個別ステップ（上級者向け）

```bash
dsmaker init "MySinger" --lang ja     # プロジェクト初期化
dsmaker slice                          # 音声をセグメントに分割
dsmaker transcribe                     # 歌詞→音素変換
dsmaker label                          # SOFA自動アライメント
dsmaker prepare                        # DiffSinger用データセット生成
dsmaker train                          # モデル学習
dsmaker export                         # ONNX出力
dsmaker package --zip                  # OpenUtauパッケージ作成
```

### Google Colabで学習する場合

```bash
dsmaker colab-export
```

生成されたzipとノートブックをColabにアップロードして実行してください。

## 対応言語

| 言語 | 歌詞入力 | 備考 |
|------|----------|------|
| 日本語 (`ja`) | 漢字・ひらがな・カタカナ対応 | pykakasi内蔵で漢字自動変換 |
| English (`en`) | そのまま英語テキスト | g2p_en必要: `pip install "dsmaker[en]"` |
| 中文 (`zh`) | 漢字そのまま | pypinyin必要: `pip install "dsmaker[zh]"` |

## パイプライン概要

```
音声ファイル + 歌詞
    ↓ スライス (無音検出で自動分割)
    ↓ G2P (歌詞→音素変換)
    ↓ SOFA (音素の時間位置を自動推定)
    ↓ データセット生成 (transcriptions.csv)
    ↓ DiffSinger学習 (音響モデル + 分散モデル)
    ↓ ONNX出力
    ↓ パッケージ
OpenUtau用歌声モデル (.zip)
```

## ライセンス

MIT
