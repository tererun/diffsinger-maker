"""Gradio Web UI for dsmaker."""

from __future__ import annotations

import shutil
import tempfile
import traceback
from pathlib import Path

import gradio as gr

from dsmaker.config import ProjectConfig
from dsmaker.utils.platform import ensure_dir, has_cuda


# ── State helpers ──

def _init_project(
    name: str, lang: str, audio_files: list[str], lyrics_text: str, backend: str,
) -> tuple[str, str]:
    """Step 0: Create project and copy files in."""
    if not name.strip():
        raise gr.Error("歌手名を入力してください")
    if not audio_files:
        raise gr.Error("音声ファイルをアップロードしてください")
    if not lyrics_text.strip():
        raise gr.Error("歌詞を入力してください")

    project_dir = Path(tempfile.mkdtemp(prefix="dsmaker_")) / name
    project_dir.mkdir(parents=True)

    config = ProjectConfig(name=name, language=lang)
    config.training.backend = backend

    songs_dir = ensure_dir(project_dir / "raw" / "songs")
    lyrics_dir = ensure_dir(project_dir / "raw" / "lyrics")
    ensure_dir(project_dir / "segments")
    ensure_dir(project_dir / "dataset" / "raw" / "wavs")
    ensure_dir(project_dir / "training")
    ensure_dir(project_dir / "export")
    ensure_dir(project_dir / "output")

    for f in audio_files:
        src = Path(f)
        shutil.copy2(src, songs_dir / src.name)

    lyrics_path = lyrics_dir / "lyrics.txt"
    lyrics_path.write_text(lyrics_text, encoding="utf-8")

    config.save(project_dir)
    return str(project_dir), f"Project initialized: {project_dir}"


def _run_slice(project_dir_str: str) -> tuple[str, str]:
    """Step 1: Slice audio."""
    from dsmaker.steps.slice import run_slice

    project_dir = Path(project_dir_str)
    config = ProjectConfig.load(project_dir)
    segments = run_slice(project_dir, config)
    result = f"Created {len(segments)} segments:\n"
    for s in segments:
        result += f"  - {s.name}\n"
    return project_dir_str, result


def _run_transcribe(project_dir_str: str) -> tuple[str, str, str]:
    """Step 2: Transcribe lyrics to phonemes."""
    from dsmaker.steps.transcribe import run_transcribe

    project_dir = Path(project_dir_str)
    config = ProjectConfig.load(project_dir)
    lab_files = run_transcribe(project_dir, config, interactive=False)

    # Build preview table
    lines = []
    for lab in lab_files:
        content = lab.read_text(encoding="utf-8")
        lines.append(f"**{lab.stem}**: `{content}`")
    preview = "\n\n".join(lines)

    # Build editable text (one lab file per line: filename<tab>phonemes)
    editable = ""
    for lab in lab_files:
        content = lab.read_text(encoding="utf-8")
        editable += f"{lab.stem}\t{content}\n"

    return project_dir_str, preview, editable


def _save_phoneme_edits(project_dir_str: str, editable_text: str) -> str:
    """Save user edits to .lab files."""
    project_dir = Path(project_dir_str)
    segments_dir = project_dir / "segments"
    count = 0
    for line in editable_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        stem, phonemes = parts
        lab_path = segments_dir / f"{stem}.lab"
        lab_path.write_text(phonemes.strip(), encoding="utf-8")
        count += 1
    return f"Saved {count} .lab files"


def _run_label(project_dir_str: str) -> tuple[str, str]:
    """Step 3: SOFA alignment."""
    from dsmaker.steps.label import run_label

    project_dir = Path(project_dir_str)
    config = ProjectConfig.load(project_dir)
    textgrids = run_label(project_dir, config)
    return project_dir_str, f"Created {len(textgrids)} TextGrid files"


def _run_prepare(project_dir_str: str) -> tuple[str, str, str | None]:
    """Step 4: Prepare dataset."""
    from dsmaker.steps.prepare import run_prepare

    project_dir = Path(project_dir_str)
    config = ProjectConfig.load(project_dir)
    run_prepare(project_dir, config)

    csv_path = project_dir / "dataset" / "raw" / "transcriptions.csv"
    csv_content = ""
    if csv_path.exists():
        csv_content = csv_path.read_text(encoding="utf-8")
    return project_dir_str, f"Dataset prepared:\n```\n{csv_content[:2000]}\n```", str(csv_path) if csv_path.exists() else None


def _run_train(project_dir_str: str) -> tuple[str, str]:
    """Step 5: Train models."""
    from dsmaker.steps.train import run_train, export_for_colab

    project_dir = Path(project_dir_str)
    config = ProjectConfig.load(project_dir)

    if config.training.backend == "colab":
        zip_path = export_for_colab(project_dir, config)
        return project_dir_str, (
            f"Colab export complete!\n\n"
            f"Files:\n"
            f"  - {zip_path}\n"
            f"  - {project_dir / 'export' / f'{config.name}_train.ipynb'}\n\n"
            f"1. Open the notebook in Google Colab\n"
            f"2. Upload the zip when prompted\n"
            f"3. Run all cells"
        )

    run_train(project_dir, config, target="both")
    return project_dir_str, "Training complete!"


def _run_export(project_dir_str: str) -> tuple[str, str]:
    """Step 6: Export ONNX."""
    from dsmaker.steps.export import run_export

    project_dir = Path(project_dir_str)
    config = ProjectConfig.load(project_dir)
    exported = run_export(project_dir, config)
    files = "\n".join(f"  - {name}: {path}" for name, path in exported.items())
    return project_dir_str, f"Exported {len(exported)} ONNX files:\n{files}"


def _run_package(project_dir_str: str) -> tuple[str, str, str | None]:
    """Step 7: Package for OpenUtau."""
    from dsmaker.steps.package import run_package

    project_dir = Path(project_dir_str)
    config = ProjectConfig.load(project_dir)
    output = run_package(project_dir, config, make_zip=True)

    zip_path = project_dir / "output" / f"{config.name}.zip"
    result = f"Voice bank packaged: {output}\n\n"
    if zip_path.exists():
        result += f"Download the zip and extract to OpenUtau's Singers folder."
    return project_dir_str, result, str(zip_path) if zip_path.exists() else None


# ── Build UI ──

def create_ui() -> gr.Blocks:
    with gr.Blocks(
        title="dsmaker — DiffSinger Voice Model Builder",
    ) as app:
        # Hidden state
        project_dir_state = gr.State("")

        gr.Markdown("# dsmaker — DiffSinger Voice Model Builder")
        gr.Markdown("音声ファイルと歌詞を入れるだけで、OpenUtau用の歌声モデルを作成できます。")

        # ── Step 0: Setup ──
        with gr.Accordion("Step 0: プロジェクト設定", open=True) as step0:
            with gr.Row():
                with gr.Column(scale=2):
                    name_input = gr.Textbox(
                        label="歌手名",
                        placeholder="MySinger",
                        value="MySinger",
                    )
                    lang_input = gr.Dropdown(
                        label="言語",
                        choices=[("日本語", "ja"), ("English", "en"), ("中文", "zh")],
                        value="ja",
                    )
                    backend_input = gr.Radio(
                        label="学習バックエンド",
                        choices=[("ローカルGPU", "local"), ("Google Colab", "colab")],
                        value="local",
                    )
                with gr.Column(scale=3):
                    audio_input = gr.File(
                        label="音声ファイル (WAV/FLAC)",
                        file_count="multiple",
                        file_types=[".wav", ".flac", ".mp3", ".ogg"],
                    )
                    lyrics_input = gr.Textbox(
                        label="歌詞（1行 = スライス後の1セグメントに対応）",
                        placeholder="さくらさくら\nやよいのそらは\nみわたすかぎり",
                        lines=8,
                    )
            init_btn = gr.Button("プロジェクト作成", variant="primary", size="lg")
            init_output = gr.Markdown()

        # ── Step 1: Slice ──
        with gr.Accordion("Step 1: 音声スライス", open=False) as step1:
            gr.Markdown("録音を無音部分で自動的にセグメントに分割します。")
            slice_btn = gr.Button("スライス実行", variant="primary")
            slice_output = gr.Markdown()

        # ── Step 2: Transcribe ──
        with gr.Accordion("Step 2: 歌詞→音素変換", open=False) as step2:
            gr.Markdown("歌詞テキストを音素列に変換します。変換結果を確認・編集できます。")
            transcribe_btn = gr.Button("音素変換実行", variant="primary")
            transcribe_preview = gr.Markdown(label="変換結果プレビュー")
            with gr.Accordion("音素を手動編集（上級者向け）", open=False):
                gr.Markdown("各行: `セグメント名<Tab>音素列` の形式で編集してください。")
                phoneme_editor = gr.Textbox(
                    label="音素エディタ",
                    lines=10,
                    interactive=True,
                )
                save_phonemes_btn = gr.Button("編集を保存")
                save_phonemes_output = gr.Markdown()

        # ── Step 3: Label ──
        with gr.Accordion("Step 3: SOFA自動アライメント", open=False) as step3:
            gr.Markdown(
                "SOFAで音素の時間位置を自動推定します。\n\n"
                "**初回実行時**: SOFAリポジトリの自動クローンと、言語モデル(.ckpt)の配置が必要です。\n"
                "モデルは `~/.cache/dsmaker/sofa_models/` に置いてください。"
            )
            label_btn = gr.Button("アライメント実行", variant="primary")
            label_output = gr.Markdown()

        # ── Step 4: Prepare ──
        with gr.Accordion("Step 4: データセット生成", open=False) as step4:
            gr.Markdown("アライメント結果からDiffSinger学習用データセットを生成します。")
            prepare_btn = gr.Button("データセット生成", variant="primary")
            prepare_output = gr.Markdown()
            prepare_download = gr.File(label="transcriptions.csv", visible=False)

        # ── Step 5: Train ──
        with gr.Accordion("Step 5: モデル学習", open=False) as step5:
            gr.Markdown(
                "DiffSingerでモデルを学習します。\n"
                "- **ローカル**: NVIDIA GPUが必要。数時間〜数日かかります。\n"
                "- **Colab**: zip + ノートブックが生成されます。Colabにアップロードして実行してください。"
            )
            train_btn = gr.Button("学習開始", variant="primary")
            train_output = gr.Markdown()

        # ── Step 6: Export ──
        with gr.Accordion("Step 6: ONNX出力", open=False) as step6:
            gr.Markdown("学習済みモデルをONNX形式でエクスポートします。")
            export_btn = gr.Button("ONNXエクスポート", variant="primary")
            export_output = gr.Markdown()

        # ── Step 7: Package ──
        with gr.Accordion("Step 7: OpenUtauパッケージ", open=False) as step7:
            gr.Markdown("モデルをOpenUtauで使える形式にパッケージします。")
            package_btn = gr.Button("パッケージ作成", variant="primary")
            package_output = gr.Markdown()
            package_download = gr.File(label="ダウンロード", visible=False)

        # ── Event wiring ──

        def on_init(name, lang, audio_files, lyrics, backend):
            if audio_files is None:
                audio_files = []
            file_paths = [f.name if hasattr(f, 'name') else str(f) for f in audio_files]
            try:
                pd, msg = _init_project(name, lang, file_paths, lyrics, backend)
                return pd, f"✅ {msg}"
            except Exception as e:
                return "", f"❌ Error: {e}"

        init_btn.click(
            fn=on_init,
            inputs=[name_input, lang_input, audio_input, lyrics_input, backend_input],
            outputs=[project_dir_state, init_output],
        )

        def on_slice(pd):
            if not pd:
                return pd, "❌ 先にプロジェクトを作成してください"
            try:
                pd, msg = _run_slice(pd)
                return pd, f"✅ {msg}"
            except Exception as e:
                return pd, f"❌ {e}\n```\n{traceback.format_exc()}\n```"

        slice_btn.click(
            fn=on_slice,
            inputs=[project_dir_state],
            outputs=[project_dir_state, slice_output],
        )

        def on_transcribe(pd):
            if not pd:
                return pd, "❌ 先にスライスを実行してください", ""
            try:
                pd, preview, editable = _run_transcribe(pd)
                return pd, f"✅ 変換完了\n\n{preview}", editable
            except Exception as e:
                return pd, f"❌ {e}\n```\n{traceback.format_exc()}\n```", ""

        transcribe_btn.click(
            fn=on_transcribe,
            inputs=[project_dir_state],
            outputs=[project_dir_state, transcribe_preview, phoneme_editor],
        )

        def on_save_phonemes(pd, editable):
            if not pd:
                return "❌ プロジェクトが未設定です"
            try:
                msg = _save_phoneme_edits(pd, editable)
                return f"✅ {msg}"
            except Exception as e:
                return f"❌ {e}"

        save_phonemes_btn.click(
            fn=on_save_phonemes,
            inputs=[project_dir_state, phoneme_editor],
            outputs=[save_phonemes_output],
        )

        def on_label(pd):
            if not pd:
                return pd, "❌ 先に音素変換を実行してください"
            try:
                pd, msg = _run_label(pd)
                return pd, f"✅ {msg}"
            except Exception as e:
                return pd, f"❌ {e}\n```\n{traceback.format_exc()}\n```"

        label_btn.click(
            fn=on_label,
            inputs=[project_dir_state],
            outputs=[project_dir_state, label_output],
        )

        def on_prepare(pd):
            if not pd:
                return pd, "❌ 先にアライメントを実行してください", gr.update(visible=False)
            try:
                pd, msg, csv_path = _run_prepare(pd)
                if csv_path:
                    return pd, f"✅ {msg}", gr.update(value=csv_path, visible=True)
                return pd, f"✅ {msg}", gr.update(visible=False)
            except Exception as e:
                return pd, f"❌ {e}\n```\n{traceback.format_exc()}\n```", gr.update(visible=False)

        prepare_btn.click(
            fn=on_prepare,
            inputs=[project_dir_state],
            outputs=[project_dir_state, prepare_output, prepare_download],
        )

        def on_train(pd):
            if not pd:
                return pd, "❌ 先にデータセットを生成してください"
            try:
                pd, msg = _run_train(pd)
                return pd, f"✅ {msg}"
            except Exception as e:
                return pd, f"❌ {e}\n```\n{traceback.format_exc()}\n```"

        train_btn.click(
            fn=on_train,
            inputs=[project_dir_state],
            outputs=[project_dir_state, train_output],
        )

        def on_export(pd):
            if not pd:
                return pd, "❌ 先に学習を実行してください"
            try:
                pd, msg = _run_export(pd)
                return pd, f"✅ {msg}"
            except Exception as e:
                return pd, f"❌ {e}\n```\n{traceback.format_exc()}\n```"

        export_btn.click(
            fn=on_export,
            inputs=[project_dir_state],
            outputs=[project_dir_state, export_output],
        )

        def on_package(pd):
            if not pd:
                return pd, "❌ 先にONNXエクスポートを実行してください", gr.update(visible=False)
            try:
                pd, msg, zip_path = _run_package(pd)
                if zip_path:
                    return pd, f"✅ {msg}", gr.update(value=zip_path, visible=True)
                return pd, f"✅ {msg}", gr.update(visible=False)
            except Exception as e:
                return pd, f"❌ {e}\n```\n{traceback.format_exc()}\n```", gr.update(visible=False)

        package_btn.click(
            fn=on_package,
            inputs=[project_dir_state],
            outputs=[project_dir_state, package_output, package_download],
        )

    return app


def launch_ui(host: str = "127.0.0.1", port: int = 7860, share: bool = False):
    app = create_ui()
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        theme=gr.themes.Soft(),
    )
