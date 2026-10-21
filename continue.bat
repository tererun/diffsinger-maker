@echo off
chcp 65001 >nul
echo ========================================
echo  dsmaker - 途中から再開
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

:: Check git
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] git が見つかりません。
    echo https://git-scm.com/downloads からインストールしてください。
    pause
    exit /b 1
)

:: Install/Update dsmaker
echo dsmaker をインストール/更新中...
pip install --no-cache-dir --force-reinstall "diffsinger-maker[ui] @ git+https://github.com/tererun/diffsinger-maker.git" >nul 2>&1
if errorlevel 1 (
    pip install --no-cache-dir --force-reinstall "diffsinger-maker[ui] @ git+https://github.com/tererun/diffsinger-maker.git"
    if errorlevel 1 (
        echo [ERROR] インストールに失敗しました。
        pause
        exit /b 1
    )
)
echo インストール完了。
echo.

:: Ask for project folder
if "%~1"=="" (
    set /p PROJECT_DIR="プロジェクトフォルダのパスを入力してください: "
) else (
    set PROJECT_DIR=%~1
)

if not exist "%PROJECT_DIR%\dsmaker.yaml" (
    echo [ERROR] %PROJECT_DIR% に dsmaker.yaml が見つかりません。
    echo 正しいプロジェクトフォルダを指定してください。
    pause
    exit /b 1
)

echo.
echo プロジェクト: %PROJECT_DIR%
echo.

:: Detect current state
echo 状態を確認中...
set HAS_SEGMENTS=0
set HAS_LABS=0
set HAS_TEXTGRID=0
set HAS_CSV=0

if exist "%PROJECT_DIR%\segments\*.wav" set HAS_SEGMENTS=1
if exist "%PROJECT_DIR%\segments\*.lab" set HAS_LABS=1
if exist "%PROJECT_DIR%\segments\*.TextGrid" set HAS_TEXTGRID=1
if exist "%PROJECT_DIR%\dataset\raw\transcriptions.csv" set HAS_CSV=1

if %HAS_SEGMENTS%==0 (
    echo  スライス: 未完了
    echo  → まず dsmaker ui でスライスから始めてください。
    echo.
    cd /d "%PROJECT_DIR%"
    dsmaker ui
    pause
    exit /b 0
)

echo  スライス: 完了
if %HAS_LABS%==1 (echo  音素変換: 完了) else (echo  音素変換: 未完了)
if %HAS_TEXTGRID%==1 (echo  アライメント: 完了) else (echo  アライメント: 未完了)
if %HAS_CSV%==1 (echo  データセット: 完了) else (echo  データセット: 未完了)
echo.

:: Run remaining steps
cd /d "%PROJECT_DIR%"

if %HAS_LABS%==0 (
    echo ========================================
    echo  歌詞入力が必要です。UIを起動します。
    echo ========================================
    echo  ブラウザで http://127.0.0.1:7860 を開いてください。
    echo.
    dsmaker ui
    pause
    exit /b 0
)

if %HAS_TEXTGRID%==0 (
    echo ========================================
    echo  Step 3: SOFA アライメント実行中...
    echo ========================================
    dsmaker label
    if errorlevel 1 (
        echo [ERROR] アライメントに失敗しました。
        pause
        exit /b 1
    )
    echo.
)

if %HAS_CSV%==0 (
    echo ========================================
    echo  Step 4: データセット生成中...
    echo ========================================
    dsmaker prepare
    if errorlevel 1 (
        echo [ERROR] データセット生成に失敗しました。
        pause
        exit /b 1
    )
    echo.
)

echo ========================================
echo  Step 5: Colab用データ生成中...
echo ========================================
dsmaker colab-export
if errorlevel 1 (
    echo [ERROR] Colab エクスポートに失敗しました。
    pause
    exit /b 1
)

echo.
echo ========================================
echo  完了！
echo ========================================
echo.
echo 以下のファイルが生成されました:
echo   %PROJECT_DIR%\export\
echo.
echo Colab で学習する場合:
echo   1. export フォルダ内の .ipynb をGoogle Colabで開く
echo   2. _colab_data.zip をアップロード
echo   3. 全セルを実行
echo.
pause
