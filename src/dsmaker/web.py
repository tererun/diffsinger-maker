"""Gradio Web UI for dsmaker."""

from __future__ import annotations

import json
import shutil
import tempfile
import traceback
from pathlib import Path

import gradio as gr

from dsmaker.config import ProjectConfig
from dsmaker.utils.platform import ensure_dir, has_cuda


# ── State helpers ──

def _init_project(name: str, lang: str, backend: str) -> tuple[str, str]:
    if not name.strip():
        raise gr.Error("歌手名を入力してください")
    project_dir = Path(tempfile.mkdtemp(prefix="dsmaker_")) / name
    project_dir.mkdir(parents=True)
    config = ProjectConfig(name=name, language=lang)
    config.training.backend = backend
    for d in ["raw/songs", "raw/lyrics", "segments", "dataset/raw/wavs",
              "training", "export", "output"]:
        ensure_dir(project_dir / d)
    config.save(project_dir)
    return str(project_dir), f"Project initialized: {project_dir}"


def _add_song(pd_str: str, audio_file: str | None, lyrics: str, songs_json: str):
    if not pd_str:
        raise gr.Error("先にプロジェクトを作成してください")
    if not audio_file:
        raise gr.Error("音声ファイルをアップロードしてください")
    project_dir = Path(pd_str)
    src = Path(audio_file)
    shutil.copy2(src, project_dir / "raw" / "songs" / src.name)
    if lyrics.strip():
        (project_dir / "raw" / "lyrics" / f"{src.stem}.txt").write_text(
            lyrics.strip(), encoding="utf-8"
        )
    songs: list[dict] = json.loads(songs_json) if songs_json else []
    songs.append({"name": src.stem, "file": src.name})
    display = "### 登録済みの曲\n" + "".join(
        f"\n{i}. **{s['name']}** ({s['file']})" for i, s in enumerate(songs, 1)
    )
    return json.dumps(songs), display, ""


def _run_slice(pd_str: str):
    from dsmaker.steps.slice import run_slice
    project_dir = Path(pd_str)
    config = ProjectConfig.load(project_dir)
    segments = run_slice(project_dir, config)
    return pd_str, f"Created {len(segments)} segments"


def _get_segment_list(pd_str: str) -> list[dict]:
    """Get list of segments with song grouping."""
    from dsmaker.steps.transcribe import get_song_segments
    project_dir = Path(pd_str)
    segments_dir = project_dir / "segments"
    song_segs = get_song_segments(segments_dir)
    result = []
    for song, segs in song_segs.items():
        for seg in segs:
            result.append({"stem": seg.stem, "song": song, "path": str(seg)})
    return result


def _get_original_lyrics(pd_str: str) -> str:
    """Get all original lyrics grouped by song for reference."""
    project_dir = Path(pd_str)
    config = ProjectConfig.load(project_dir)
    lyrics_dir = project_dir / config.lyrics_dir
    parts = []
    for f in sorted(lyrics_dir.glob("*.txt")):
        text = f.read_text(encoding="utf-8").strip()
        parts.append(f"--- {f.stem} ---\n{text}")
    return "\n\n".join(parts) if parts else "(歌詞ファイルなし)"


def _load_segment_for_editor(pd_str: str, seg_index: int, assignments_json: str):
    """Load a specific segment for the editor: audio path, current lyrics."""
    segments = _get_segment_list(pd_str)
    if not segments or seg_index < 0 or seg_index >= len(segments):
        return None, "", f"0 / 0", ""

    seg = segments[seg_index]
    assignments: dict = json.loads(assignments_json) if assignments_json else {}
    current_lyrics = assignments.get(seg["stem"], "")
    progress = f"{seg_index + 1} / {len(segments)}"
    label = f"**[{seg['song']}]** {seg['stem']}"
    return seg["path"], current_lyrics, progress, label


def _save_and_navigate(
    pd_str: str, current_index: int, lyrics_text: str,
    assignments_json: str, direction: int,
):
    """Save current lyrics and move to next/prev segment."""
    segments = _get_segment_list(pd_str)
    assignments: dict = json.loads(assignments_json) if assignments_json else {}

    # Save current
    if segments and 0 <= current_index < len(segments):
        stem = segments[current_index]["stem"]
        assignments[stem] = lyrics_text.strip()

    # Navigate
    new_index = current_index + direction
    new_index = max(0, min(new_index, len(segments) - 1))

    new_assignments_json = json.dumps(assignments, ensure_ascii=False)

    # Load new segment
    seg = segments[new_index]
    new_lyrics = assignments.get(seg["stem"], "")
    progress = f"{new_index + 1} / {len(segments)}"
    label = f"**[{seg['song']}]** {seg['stem']}"

    return (
        new_index,
        new_assignments_json,
        seg["path"],
        new_lyrics,
        progress,
        label,
    )


def _build_assignments_summary(pd_str: str, assignments_json: str) -> str:
    """Build a summary of all segment-lyrics assignments."""
    segments = _get_segment_list(pd_str)
    assignments: dict = json.loads(assignments_json) if assignments_json else {}
    lines = []
    current_song = ""
    filled = 0
    for seg in segments:
        if seg["song"] != current_song:
            current_song = seg["song"]
            lines.append(f"\n**{current_song}**")
        lyrics = assignments.get(seg["stem"], "")
        mark = "o" if lyrics else "x"
        filled += 1 if lyrics else 0
        display = lyrics if lyrics else "(未入力)"
        lines.append(f"- [{mark}] {seg['stem']}: {display}")
    header = f"### 入力状況: {filled}/{len(segments)} セグメント\n"
    return header + "\n".join(lines)


def _run_transcribe_from_assignments(pd_str: str, assignments_json: str):
    """Convert all assigned lyrics to phonemes."""
    from dsmaker.steps.transcribe import run_transcribe, get_song_segments

    project_dir = Path(pd_str)
    config = ProjectConfig.load(project_dir)
    assignments: dict = json.loads(assignments_json) if assignments_json else {}

    segment_lyrics = [(stem, lyrics) for stem, lyrics in assignments.items()]

    # Save lyrics files for consistency
    segments_dir = project_dir / "segments"
    lyrics_dir = project_dir / config.lyrics_dir
    song_segs = get_song_segments(segments_dir)
    for song_stem, segs in song_segs.items():
        lines = [assignments.get(seg.stem, "") for seg in segs]
        (lyrics_dir / f"{song_stem}.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    lab_files = run_transcribe(
        project_dir, config, interactive=False, segment_lyrics=segment_lyrics
    )

    preview_lines = []
    for lab in lab_files:
        content = lab.read_text(encoding="utf-8")
        preview_lines.append(f"**{lab.stem}**: `{content}`")

    phoneme_edit = "".join(
        f"{lab.stem}\t{lab.read_text(encoding='utf-8')}\n" for lab in lab_files
    )
    return pd_str, "\n\n".join(preview_lines), phoneme_edit


def _save_phoneme_edits(pd_str: str, editable_text: str) -> str:
    project_dir = Path(pd_str)
    segments_dir = project_dir / "segments"
    count = 0
    for line in editable_text.strip().splitlines():
        parts = line.strip().split("\t", 1)
        if len(parts) == 2:
            (segments_dir / f"{parts[0]}.lab").write_text(
                parts[1].strip(), encoding="utf-8"
            )
            count += 1
    return f"Saved {count} .lab files"


def _run_label(pd_str: str):
    from dsmaker.steps.label import run_label
    project_dir = Path(pd_str)
    config = ProjectConfig.load(project_dir)
    tg = run_label(project_dir, config)
    return pd_str, f"Created {len(tg)} TextGrid files"


def _run_prepare(pd_str: str):
    from dsmaker.steps.prepare import run_prepare
    project_dir = Path(pd_str)
    config = ProjectConfig.load(project_dir)
    run_prepare(project_dir, config)
    csv_path = project_dir / "dataset" / "raw" / "transcriptions.csv"
    csv_content = csv_path.read_text(encoding="utf-8") if csv_path.exists() else ""
    return (
        pd_str,
        f"Dataset prepared:\n```\n{csv_content[:2000]}\n```",
        str(csv_path) if csv_path.exists() else None,
    )


def _run_train(pd_str: str):
    from dsmaker.steps.train import run_train, export_for_colab
    project_dir = Path(pd_str)
    config = ProjectConfig.load(project_dir)
    if config.training.backend == "colab":
        zip_path = export_for_colab(project_dir, config)
        return pd_str, (
            f"Colab export complete!\n\n"
            f"  - {zip_path}\n"
            f"  - {project_dir / 'export' / f'{config.name}_train.ipynb'}\n\n"
            f"1. Open the notebook in Google Colab\n"
            f"2. Upload the zip when prompted\n"
            f"3. Run all cells"
        )
    run_train(project_dir, config, target="both")
    return pd_str, "Training complete!"


def _run_export(pd_str: str):
    from dsmaker.steps.export import run_export
    project_dir = Path(pd_str)
    config = ProjectConfig.load(project_dir)
    exported = run_export(project_dir, config)
    files = "\n".join(f"  - {n}: {p}" for n, p in exported.items())
    return pd_str, f"Exported {len(exported)} ONNX files:\n{files}"


def _run_package(pd_str: str):
    from dsmaker.steps.package import run_package
    project_dir = Path(pd_str)
    config = ProjectConfig.load(project_dir)
    output = run_package(project_dir, config, make_zip=True)
    zip_path = project_dir / "output" / f"{config.name}.zip"
    msg = f"Voice bank packaged: {output}\n\n"
    if zip_path.exists():
        msg += "Download the zip and extract to OpenUtau's Singers folder."
    return pd_str, msg, str(zip_path) if zip_path.exists() else None


# ── Build UI ──

def create_ui() -> gr.Blocks:
    with gr.Blocks(title="dsmaker — DiffSinger Voice Model Builder") as app:

        # ── State ──
        project_dir_state = gr.State("")
        songs_state = gr.State("")
        assignments_state = gr.State("{}")  # {segment_stem: lyrics}
        seg_index_state = gr.State(0)

        gr.Markdown("# dsmaker — DiffSinger Voice Model Builder")
        gr.Markdown("音声ファイルと歌詞を入れるだけで、OpenUtau用の歌声モデルを作成できます。")

        # ── Step 0: Project setup ──
        with gr.Accordion("Step 0: プロジェクト設定", open=True):
            with gr.Row():
                name_input = gr.Textbox(label="歌手名", value="MySinger", scale=2)
                lang_input = gr.Dropdown(
                    label="言語",
                    choices=[("日本語", "ja"), ("English", "en"), ("中文", "zh")],
                    value="ja", scale=1,
                )
                backend_input = gr.Radio(
                    label="学習バックエンド",
                    choices=[("ローカルGPU", "local"), ("Google Colab", "colab")],
                    value="local", scale=2,
                )
            init_btn = gr.Button("プロジェクト作成", variant="primary")
            init_output = gr.Markdown()

        # ── Step 0.5: Add songs ──
        with gr.Accordion("Step 0.5: 曲を追加", open=False):
            gr.Markdown("曲を1つずつ追加。各曲にWAVとその歌詞全文を入れます。")
            with gr.Row():
                song_audio = gr.File(
                    label="音声ファイル (1曲)", file_count="single",
                    file_types=[".wav", ".flac", ".mp3", ".ogg"], scale=1,
                )
                song_lyrics = gr.Textbox(
                    label="この曲の歌詞（全文・参照用）",
                    placeholder="桜咲く春の空に\n風が優しく吹いて",
                    lines=6, scale=2,
                )
            add_song_btn = gr.Button("この曲を追加", variant="secondary", size="lg")
            songs_display = gr.Markdown("まだ曲が追加されていません。")

        # ── Step 1: Slice ──
        with gr.Accordion("Step 1: 音声スライス", open=False):
            gr.Markdown("全ての曲を無音部分で自動分割します。")
            slice_btn = gr.Button("スライス実行", variant="primary")
            slice_output = gr.Markdown()

        # ── Step 2: Segment lyrics editor ──
        with gr.Accordion("Step 2: セグメント歌詞入力", open=False):
            gr.Markdown(
                "各セグメントを再生して聴き、対応する歌詞を入力してください。\n"
                "右側に元の歌詞全文が表示されるので参照しながら入力できます。"
            )

            with gr.Row():
                # Left: segment editor
                with gr.Column(scale=3):
                    seg_label = gr.Markdown("**セグメントを選択してください**")
                    seg_progress = gr.Textbox(
                        label="進捗", value="0 / 0", interactive=False,
                        max_lines=1,
                    )
                    seg_audio = gr.Audio(
                        label="セグメント再生", type="filepath", interactive=False,
                    )
                    seg_lyrics_input = gr.Textbox(
                        label="このセグメントの歌詞",
                        placeholder="ここに歌詞を入力...",
                        lines=2,
                    )
                    with gr.Row():
                        prev_btn = gr.Button("← 前へ", size="sm", scale=1)
                        save_seg_btn = gr.Button(
                            "保存して次へ →", variant="primary", size="sm", scale=2,
                        )
                        next_btn = gr.Button("次へ →", size="sm", scale=1)

                # Right: original lyrics reference + assignment summary
                with gr.Column(scale=2):
                    original_lyrics_display = gr.Textbox(
                        label="元の歌詞（参照用）",
                        interactive=False,
                        lines=12,
                    )
                    assignment_summary = gr.Markdown("### 入力状況")

            gr.Markdown("---")
            transcribe_btn = gr.Button(
                "全セグメントの音素変換を実行", variant="primary", size="lg",
            )
            transcribe_preview = gr.Markdown()

            with gr.Accordion("音素を手動編集（上級者向け）", open=False):
                phoneme_editor = gr.Textbox(label="音素エディタ", lines=10, interactive=True)
                save_phonemes_btn = gr.Button("編集を保存")
                save_phonemes_output = gr.Markdown()

        # ── Steps 3-7 (unchanged) ──
        with gr.Accordion("Step 3: SOFA自動アライメント", open=False):
            gr.Markdown(
                "SOFAで音素の時間位置を自動推定します。\n\n"
                "**初回**: `~/.cache/dsmaker/sofa_models/` にSOFAモデル(.ckpt)を配置してください。"
            )
            label_btn = gr.Button("アライメント実行", variant="primary")
            label_output = gr.Markdown()

        with gr.Accordion("Step 4: データセット生成", open=False):
            prepare_btn = gr.Button("データセット生成", variant="primary")
            prepare_output = gr.Markdown()
            prepare_download = gr.File(label="transcriptions.csv", visible=False)

        with gr.Accordion("Step 5: モデル学習", open=False):
            gr.Markdown(
                "- **ローカル**: NVIDIA GPU必要。数時間〜数日。\n"
                "- **Colab**: zip + ノートブック生成 → Colabで実行。"
            )
            train_btn = gr.Button("学習開始", variant="primary")
            train_output = gr.Markdown()

        with gr.Accordion("Step 6: ONNX出力", open=False):
            export_btn = gr.Button("ONNXエクスポート", variant="primary")
            export_output = gr.Markdown()

        with gr.Accordion("Step 7: OpenUtauパッケージ", open=False):
            package_btn = gr.Button("パッケージ作成", variant="primary")
            package_output = gr.Markdown()
            package_download = gr.File(label="ダウンロード", visible=False)

        # ── Event wiring ──

        # Step 0: Init
        def on_init(name, lang, backend):
            try:
                pd, msg = _init_project(name, lang, backend)
                return pd, f"✅ {msg}", "", "{}", 0
            except Exception as e:
                return "", f"❌ {e}", "", "{}", 0

        init_btn.click(
            fn=on_init,
            inputs=[name_input, lang_input, backend_input],
            outputs=[project_dir_state, init_output, songs_state, assignments_state, seg_index_state],
        )

        # Step 0.5: Add song
        def on_add_song(pd, audio, lyrics, songs_json):
            if audio is None:
                return songs_json, "❌ 音声ファイルをアップロードしてください", ""
            file_path = audio.name if hasattr(audio, "name") else str(audio)
            try:
                new_songs, display, _ = _add_song(pd, file_path, lyrics, songs_json)
                return new_songs, f"✅ 曲を追加しました\n\n{display}", ""
            except Exception as e:
                return songs_json, f"❌ {e}", ""

        add_song_btn.click(
            fn=on_add_song,
            inputs=[project_dir_state, song_audio, song_lyrics, songs_state],
            outputs=[songs_state, songs_display, song_lyrics],
        )

        # Step 1: Slice → then load first segment into editor
        def on_slice(pd):
            if not pd:
                return pd, "❌ 先にプロジェクトを作成してください", "{}", 0, None, "", "0 / 0", "", "", ""
            try:
                pd, msg = _run_slice(pd)
                # Load first segment
                segments = _get_segment_list(pd)
                orig_lyrics = _get_original_lyrics(pd)
                if segments:
                    seg = segments[0]
                    progress = f"1 / {len(segments)}"
                    label = f"**[{seg['song']}]** {seg['stem']}"
                    summary = _build_assignments_summary(pd, "{}")
                    return pd, f"✅ {msg}", "{}", 0, seg["path"], "", progress, label, orig_lyrics, summary
                return pd, f"✅ {msg}", "{}", 0, None, "", "0 / 0", "", orig_lyrics, ""
            except Exception as e:
                return pd, f"❌ {e}\n```\n{traceback.format_exc()}\n```", "{}", 0, None, "", "0 / 0", "", "", ""

        slice_btn.click(
            fn=on_slice,
            inputs=[project_dir_state],
            outputs=[
                project_dir_state, slice_output,
                assignments_state, seg_index_state,
                seg_audio, seg_lyrics_input, seg_progress, seg_label,
                original_lyrics_display, assignment_summary,
            ],
        )

        # Navigation: prev / next / save+next
        def _navigate(pd, idx, lyrics, assignments_json, direction):
            if not pd:
                return idx, assignments_json, None, "", "0 / 0", "", ""
            try:
                new_idx, new_assign, audio, new_lyrics, progress, label = \
                    _save_and_navigate(pd, idx, lyrics, assignments_json, direction)
                summary = _build_assignments_summary(pd, new_assign)
                return new_idx, new_assign, audio, new_lyrics, progress, label, summary
            except Exception as e:
                return idx, assignments_json, None, "", "0 / 0", f"❌ {e}", ""

        nav_outputs = [
            seg_index_state, assignments_state,
            seg_audio, seg_lyrics_input, seg_progress, seg_label,
            assignment_summary,
        ]
        nav_inputs = [project_dir_state, seg_index_state, seg_lyrics_input, assignments_state]

        prev_btn.click(
            fn=lambda pd, idx, ly, aj: _navigate(pd, idx, ly, aj, -1),
            inputs=nav_inputs, outputs=nav_outputs,
        )
        next_btn.click(
            fn=lambda pd, idx, ly, aj: _navigate(pd, idx, ly, aj, 1),
            inputs=nav_inputs, outputs=nav_outputs,
        )
        save_seg_btn.click(
            fn=lambda pd, idx, ly, aj: _navigate(pd, idx, ly, aj, 1),
            inputs=nav_inputs, outputs=nav_outputs,
        )

        # Step 2: Transcribe
        def on_transcribe(pd, idx, lyrics, assignments_json):
            if not pd:
                return pd, "❌ 先にスライスを実行してください", ""
            # Save current segment first
            segments = _get_segment_list(pd)
            assignments: dict = json.loads(assignments_json) if assignments_json else {}
            if segments and 0 <= idx < len(segments):
                assignments[segments[idx]["stem"]] = lyrics.strip()
            try:
                pd, preview, phoneme_edit = _run_transcribe_from_assignments(
                    pd, json.dumps(assignments, ensure_ascii=False)
                )
                return pd, f"✅ 変換完了\n\n{preview}", phoneme_edit
            except Exception as e:
                return pd, f"❌ {e}\n```\n{traceback.format_exc()}\n```", ""

        transcribe_btn.click(
            fn=on_transcribe,
            inputs=[project_dir_state, seg_index_state, seg_lyrics_input, assignments_state],
            outputs=[project_dir_state, transcribe_preview, phoneme_editor],
        )

        # Save phonemes
        def on_save_phonemes(pd, editable):
            if not pd:
                return "❌ プロジェクトが未設定です"
            try:
                return f"✅ {_save_phoneme_edits(pd, editable)}"
            except Exception as e:
                return f"❌ {e}"

        save_phonemes_btn.click(
            fn=on_save_phonemes,
            inputs=[project_dir_state, phoneme_editor],
            outputs=[save_phonemes_output],
        )

        # Steps 3-7: simple wrappers
        def _wrap(fn, *extra_out):
            def handler(pd):
                if not pd:
                    return (pd, "❌ プロジェクトが未設定です") + tuple(
                        gr.update(visible=False) for _ in extra_out
                    )
                try:
                    result = fn(pd)
                    if len(extra_out) == 0:
                        return result[0], f"✅ {result[1]}"
                    out = [result[0], f"✅ {result[1]}"]
                    for i, _ in enumerate(extra_out):
                        val = result[2 + i] if 2 + i < len(result) else None
                        out.append(
                            gr.update(value=val, visible=True) if val
                            else gr.update(visible=False)
                        )
                    return tuple(out)
                except Exception as e:
                    out = [pd, f"❌ {e}\n```\n{traceback.format_exc()}\n```"]
                    out += [gr.update(visible=False) for _ in extra_out]
                    return tuple(out)
            return handler

        label_btn.click(
            fn=_wrap(_run_label),
            inputs=[project_dir_state],
            outputs=[project_dir_state, label_output],
        )
        prepare_btn.click(
            fn=_wrap(_run_prepare, "file"),
            inputs=[project_dir_state],
            outputs=[project_dir_state, prepare_output, prepare_download],
        )
        train_btn.click(
            fn=_wrap(_run_train),
            inputs=[project_dir_state],
            outputs=[project_dir_state, train_output],
        )
        export_btn.click(
            fn=_wrap(_run_export),
            inputs=[project_dir_state],
            outputs=[project_dir_state, export_output],
        )
        package_btn.click(
            fn=_wrap(_run_package, "file"),
            inputs=[project_dir_state],
            outputs=[project_dir_state, package_output, package_download],
        )

    return app


def launch_ui(host: str = "127.0.0.1", port: int = 7860, share: bool = False):
    app = create_ui()
    app.launch(server_name=host, server_port=port, share=share, theme=gr.themes.Soft())
