# app.py
import os
import io
import time
import shutil
import pathlib
import traceback
from typing import Optional, Tuple, List

import streamlit as st
import pandas as pd

# OpenAI SDK (official)
try:
    from openai import OpenAI
except ImportError:
    st.stop()

# -----------------------------
# Utilities
# -----------------------------
SUPPORTED_TABLE_EXTS = (".xlsx", ".xls", ".csv")

def get_openai_client(api_key: str) -> OpenAI:
    if not api_key:
        st.error("⚠️ API Key is required. Please enter your OpenAI API Key above.")
        st.stop()
    return OpenAI(api_key=api_key)


def discover_files(folder: str) -> List[pathlib.Path]:
    p = pathlib.Path(folder)
    files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_TABLE_EXTS]
    # Stable & predictable order
    files.sort(key=lambda x: (x.stat().st_mtime, x.name))
    return files

def safe_basename_no_ext(path: pathlib.Path) -> str:
    # Use base name without extension
    return path.stem

def safe_slug(s: str, max_len: int = 60) -> str:
    # Keep Japanese, alnum, _, -, and space → _
    import re
    s = s.strip()
    s = re.sub(r"[^\w\- \u3040-\u30FF\u4E00-\u9FFF\uFF01-\uFF60]", "_", s)  # allow JP ranges
    s = re.sub(r"\s+", "_", s)
    return s[:max_len] if len(s) > max_len else s

def ensure_subfolder_for_file(src_file: pathlib.Path) -> pathlib.Path:
    parent = src_file.parent
    sub = parent / safe_basename_no_ext(src_file)
    sub.mkdir(parents=True, exist_ok=True)
    return sub

def move_file_to_folder(src_file: pathlib.Path, dest_folder: pathlib.Path) -> pathlib.Path:
    dest = dest_folder / src_file.name
    if dest.exists():
        # If already moved previously, keep existing
        return dest
    shutil.move(str(src_file), str(dest))
    return dest

def read_table(path: pathlib.Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    # default to Excel
    return pd.read_excel(path)  # first sheet by default

def find_columns(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    """
    Return (script_col, scene_col_or_none).
    Looks for a 'Script' header (case-insensitive exact) and for scene number in 'Scene' or 'Scene number'.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    # Script column
    script_key = None
    for key in ["script", "本文", "セリフ", "台詞"]:
        if key in cols_lower:
            script_key = cols_lower[key]
            break
    if not script_key:
        # Try fuzzy contains 'script'
        for c in df.columns:
            if "script" in c.lower():
                script_key = c
                break
    if not script_key:
        raise ValueError("Could not find a 'Script' column. Make sure the file has a 'Script' header.")

    # Scene column (optional; used to order & name)
    scene_key = None
    for key in ["scene number", "scene", "シーン", "シーン番号"]:
        if key in cols_lower:
            scene_key = cols_lower[key]
            break
    if not scene_key:
        # Fallback: look for a numeric-ish first column
        first_col = df.columns[0]
        if pd.api.types.is_integer_dtype(df[first_col]) or pd.api.types.is_numeric_dtype(df[first_col]):
            scene_key = first_col

    return script_key, scene_key

def coerce_scene_number(val, idx: int) -> int:
    try:
        n = int(val)
        if n >= 0:
            return n
    except Exception:
        pass
    # Fallback to 1-based row index order
    return idx + 1

# def tts_openai_mp3(client: OpenAI, text: str, model: str = "gpt-4o-mini-tts", voice: str = "alloy") -> bytes:
#     """
#     Generates MP3 bytes using OpenAI TTS.
#     """
#     # Defensive trim: OpenAI TTS accepts reasonably long inputs, but we limit very long chunks
#     text = text.strip()
#     if not text:
#         return b""

#     # API call
#     resp = client.audio.speech.create(
#         model=model,
#         voice=voice,
#         input=text,
#         format="mp3",
#         # You can add "language": "ja" to be explicit, but JP text should auto-detect
#     )
#     # SDK returns .content (bytes) in recent versions; fallback to .read() for older
#     audio_bytes = getattr(resp, "content", None)
#     if audio_bytes is None and hasattr(resp, "read"):
#         audio_bytes = resp.read()
#     if audio_bytes is None:
#         raise RuntimeError("TTS response did not contain audio bytes.")
#     return audio_bytes
def tts_openai_mp3(client, text: str, model: str = "gpt-4o-mini-tts", voice: str = "alloy") -> bytes:
    """
    Generate MP3 bytes using OpenAI TTS across SDK variants.
    Tries multiple signatures to stay compatible with different openai versions.
    """
    text = (text or "").strip()
    if not text:
        return b""

    # Try 1: some SDKs accept `format="mp3"`
    try:
        resp = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            format="mp3",  # <-- may raise unexpected keyword error in some versions
        )
        audio_bytes = getattr(resp, "content", None) or (resp.read() if hasattr(resp, "read") else None)
        if not audio_bytes:
            raise RuntimeError("TTS response had no bytes (format=mp3 path).")
        return audio_bytes
    except TypeError:
        pass  # fall through to other signatures

    # Try 2: some SDKs use `response_format="mp3"`
    try:
        resp = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format="mp3",
        )
        audio_bytes = getattr(resp, "content", None) or (resp.read() if hasattr(resp, "read") else None)
        if not audio_bytes:
            raise RuntimeError("TTS response had no bytes (response_format path).")
        return audio_bytes
    except TypeError:
        pass

    # Try 3: no explicit format (many versions default to MP3)
    resp = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
    )
    audio_bytes = getattr(resp, "content", None) or (resp.read() if hasattr(resp, "read") else None)
    if not audio_bytes:
        raise RuntimeError("TTS response had no bytes (no-format path).")
    return audio_bytes

def save_bytes(path: pathlib.Path, content: bytes):
    with open(path, "wb") as f:
        f.write(content)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Excel→JP Audio Batch", page_icon="🎧", layout="centered")
st.title("Excel → 日本語音声 一括生成ツール 🎧")

st.markdown(
    "1) フォルダを指定 → 2) 各Excel/CSVをサブフォルダへ移動 → "
    "3) Script列の各行を**日本語音声(MP3)**に変換 → 4) 全ファイルを処理"
)
api_key = st.text_input("OpenAI API Key", type="password")
folder = st.text_input("処理するフォルダパス", value=os.getcwd())
tts_model = st.selectbox("TTSモデル (OpenAI)", ["gpt-4o-mini-tts", "tts-1"])
voice = st.text_input("音声(voice) 名（例: alloy, verse, etc.）", value="alloy")
zero_pad = st.number_input("シーン番号ゼロ埋め桁数", min_value=2, max_value=6, value=3, step=1)
prefix_name = st.text_input("音声ファイルの接頭辞（任意）", value="scene")

start_btn = st.button("処理を開始")

if start_btn:
    try:
        client = get_openai_client(api_key)
        files = discover_files(folder)
        if not files:
            st.warning("Excel/CSVファイルが見つかりませんでした。")
            st.stop()

        overall_progress = st.progress(0.0, text="全体進捗…")
        file_status = st.empty()

        for fi_idx, src_file in enumerate(files):
            file_status.info(f"処理中: {src_file.name}")

            # 1) サブフォルダ作成 & ファイル移動
            subfolder = ensure_subfolder_for_file(src_file)
            moved_path = move_file_to_folder(src_file, subfolder)

            # 2) 読み込み
            try:
                df = read_table(moved_path)
            except Exception as e:
                st.error(f"読み込み失敗: {moved_path.name}\n{e}")
                continue

            # 3) 列の特定
            try:
                script_col, scene_col = find_columns(df)
            except Exception as e:
                st.error(f"{moved_path.name}: {e}")
                continue

            # 4) 行ごとに音声生成
            per_file_prog = st.progress(0.0, text=f"{moved_path.name} の音声生成…")
            n_rows = len(df)
            created = 0

            for idx, row in df.iterrows():
                text = str(row[script_col]).strip() if pd.notna(row[script_col]) else ""
                if not text:
                    per_file_prog.progress((idx + 1) / max(1, n_rows), text=f"空行スキップ {idx+1}/{n_rows}")
                    continue

                # シーン番号
                scene_num = coerce_scene_number(row[scene_col], idx) if scene_col else (idx + 1)
                scene_tag = str(scene_num).zfill(zero_pad)

                # ファイル名組み立て
                short_snippet = safe_slug(text[:20]) or "line"
                out_name = f"{prefix_name}_{scene_tag}_{short_snippet}.mp3"
                out_path = subfolder / out_name
                if out_path.exists():
                    # 既存ならスキップ（重複生成防止）
                    per_file_prog.progress((idx + 1) / max(1, n_rows), text=f"既存: {out_name}  {idx+1}/{n_rows}")
                    continue

                # TTS 試行（リトライ付き）
                max_retries = 3
                for attempt in range(1, max_retries + 1):
                    try:
                        audio_bytes = tts_openai_mp3(client, text, model=tts_model, voice=voice)
                        if audio_bytes:
                            save_bytes(out_path, audio_bytes)
                            created += 1
                        break
                    except Exception as e:
                        if attempt == max_retries:
                            st.error(f"TTS失敗 ({moved_path.name} 行{idx+1}): {e}")
                        else:
                            time.sleep(1.0 * attempt)  # 指数バックオフ

                per_file_prog.progress((idx + 1) / max(1, n_rows),
                                       text=f"生成 {created} 件 / {idx+1}/{n_rows}")

            per_file_prog.empty()
            st.success(f"完了: {moved_path.name} → {created} 件の音声を作成。保存先: {subfolder}")

            overall_progress.progress((fi_idx + 1) / max(1, len(files)),
                                      text=f"全体進捗 {fi_idx+1}/{len(files)}")

        overall_progress.empty()
        file_status.empty()
        st.balloons()
        st.success("すべてのファイルを処理しました。")

        st.markdown("**フォルダ構成**：各ファイル名（拡張子除く）と同名のサブフォルダに、元Excel/CSVと音声MP3を格納します。")
    except Exception as e:
        st.error("予期せぬエラーが発生しました。")
        st.code("".join(traceback.format_exc()))
