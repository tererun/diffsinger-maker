#!/bin/bash
set -e

echo "========================================"
echo " dsmaker - DiffSinger Voice Model Builder"
echo "========================================"
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 が見つかりません。"
    echo "  Mac: brew install python3"
    echo "  Linux: sudo apt install python3 python3-pip"
    exit 1
fi

PYVER=$(python3 --version)
echo "$PYVER が見つかりました。"

# Install/Update dsmaker
echo
echo "dsmaker をインストール中..."
pip3 install --no-cache-dir --force-reinstall "diffsinger-maker[ui] @ git+https://github.com/tererun/diffsinger-maker.git"

echo
echo "========================================"
echo " インストール完了！"
echo "========================================"
echo
echo "ブラウザで http://127.0.0.1:7860 を開いてください。"
echo "終了するには Ctrl+C を押してください。"
echo

dsmaker ui
