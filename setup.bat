@echo off
chcp 65001 >nul
echo ========================================
echo  dsmaker - DiffSinger Voice Model Builder
echo ========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python が見つかりません。
    echo https://www.python.org/downloads/ からPython 3.10以上をインストールしてください。
    echo インストール時に「Add Python to PATH」にチェックを入れてください。
    pause
    exit /b 1
)

:: Check Python version
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo Python %PYVER% が見つかりました。

:: Install/Update dsmaker
echo.
echo dsmaker をインストール中...
pip install --no-cache-dir --force-reinstall "diffsinger-maker[ui] @ git+https://github.com/tererun/diffsinger-maker.git"
if errorlevel 1 (
    echo.
    echo [ERROR] インストールに失敗しました。
    echo git がインストールされているか確認してください。
    echo https://git-scm.com/downloads からインストールできます。
    pause
    exit /b 1
)

echo.
echo ========================================
echo  インストール完了！
echo ========================================
echo.
echo ブラウザで http://127.0.0.1:7860 を開いてください。
echo 終了するには Ctrl+C を押してください。
echo.

dsmaker ui
pause
