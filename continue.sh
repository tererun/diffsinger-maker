#!/bin/bash
set -e

echo "========================================"
echo " dsmaker - 途中から再開"
echo "========================================"
echo

# Check dependencies
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 が見つかりません。"
    exit 1
fi
if ! command -v git &> /dev/null; then
    echo "[ERROR] git が見つかりません。"
    exit 1
fi

# Install/Update dsmaker
echo "dsmaker をインストール/更新中..."
pip3 install --no-cache-dir --force-reinstall "diffsinger-maker[ui] @ git+https://github.com/tererun/diffsinger-maker.git" 2>&1 | tail -1
echo

# Get project dir
PROJECT_DIR="${1:-}"
if [ -z "$PROJECT_DIR" ]; then
    read -p "プロジェクトフォルダのパスを入力してください: " PROJECT_DIR
fi

if [ ! -f "$PROJECT_DIR/dsmaker.yaml" ]; then
    echo "[ERROR] $PROJECT_DIR に dsmaker.yaml が見つかりません。"
    exit 1
fi

echo
echo "プロジェクト: $PROJECT_DIR"
echo

# Detect state
HAS_SEGMENTS=0; HAS_LABS=0; HAS_TEXTGRID=0; HAS_CSV=0
ls "$PROJECT_DIR/segments/"*.wav &>/dev/null && HAS_SEGMENTS=1
ls "$PROJECT_DIR/segments/"*.lab &>/dev/null && HAS_LABS=1
ls "$PROJECT_DIR/segments/"*.TextGrid &>/dev/null && HAS_TEXTGRID=1
[ -f "$PROJECT_DIR/dataset/raw/transcriptions.csv" ] && HAS_CSV=1

echo "状態:"
[ $HAS_SEGMENTS -eq 1 ] && echo "  スライス: 完了" || echo "  スライス: 未完了"
[ $HAS_LABS -eq 1 ] && echo "  音素変換: 完了" || echo "  音素変換: 未完了"
[ $HAS_TEXTGRID -eq 1 ] && echo "  アライメント: 完了" || echo "  アライメント: 未完了"
[ $HAS_CSV -eq 1 ] && echo "  データセット: 完了" || echo "  データセット: 未完了"
echo

cd "$PROJECT_DIR"

if [ $HAS_SEGMENTS -eq 0 ]; then
    echo "スライスが未完了です。UIを起動します。"
    dsmaker ui
    exit 0
fi

if [ $HAS_LABS -eq 0 ]; then
    echo "歌詞入力が必要です。UIを起動します。"
    dsmaker ui
    exit 0
fi

if [ $HAS_TEXTGRID -eq 0 ]; then
    echo "========================================"
    echo " Step 3: SOFA アライメント実行中..."
    echo "========================================"
    dsmaker label
    echo
fi

if [ $HAS_CSV -eq 0 ]; then
    echo "========================================"
    echo " Step 4: データセット生成中..."
    echo "========================================"
    dsmaker prepare
    echo
fi

echo "========================================"
echo " Step 5: Colab用データ生成中..."
echo "========================================"
dsmaker colab-export

echo
echo "========================================"
echo " 完了！"
echo "========================================"
echo
echo "export/ フォルダ内のファイル:"
ls -1 "$PROJECT_DIR/export/"
echo
echo "Colab で学習する場合:"
echo "  1. export/ 内の .ipynb をGoogle Colabで開く"
echo "  2. _colab_data.zip をアップロード"
echo "  3. 全セルを実行"
