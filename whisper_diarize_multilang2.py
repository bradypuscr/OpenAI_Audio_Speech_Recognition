#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Whisper + Pyannote diarization + per-segment language ID (langid)
+ Study-friendly TXT with kana/romaji for Japanese segments
-----------------------------------------------------------------
- Inputs: an audio/video file path
- Outputs: JSON / SRT / VTT / TXT / CSV / STUDY next to the input file (with a suffix)
- GPU ready: auto-detects CUDA and uses it when available
- TorchCodec path (preferred) with automatic fallback to preloaded-audio if unavailable

Tip (optional pre-clean):
  ffmpeg -i in.mp4 -vn -ac 1 -ar 16000 -af "arnndn=m=rnnoise-models/somnolent_hogwash.rnnn" clean.wav
"""

"""
--------------- ENVIRONMENT SETUP ---------------
# fresh start
conda deactivate
conda env remove --name py312 -y
conda clean --all -y

conda create -n py312 python=3.12 -y
conda activate py312

# multimedia dependencies
conda install "ffmpeg<8" -y

# ✅ install PyTorch stack with matching CUDA runtime (no full toolkit)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# diarization and audio utils
pip install pyannote.audio pykakasi faster-whisper argostranslate langid pandas tqdm

# optional: ensure bundled cuDNN takes precedence (Linux/WSL)
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/99-cudnn-path.sh" <<'SH'
# Ensure PyTorch uses the bundled cuDNN first
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
SH

# reload the env
conda deactivate && conda activate py312
"""

import os
import sys
import json
import argparse
import logging
import subprocess
import tempfile
import textwrap
import numpy as np
import time
import shlex
import re
import io
import unicodedata
import shutil
import csv
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Union, Optional

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning, module="ctranslate2",)
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
warnings.filterwarnings("ignore", module="pyannote.audio.utils.reproducibility", category=UserWarning)
warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom is <= 0")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pykakasi")
warnings.filterwarnings("ignore", message=r"Please use the new API settings to control TF32 behavior", category=UserWarning, module=r"torch\.backends\.cuda")

import langid
import pandas as pd
from tqdm import tqdm

# --- pykakasi: modern API (no deprecated calls)
try:
    import pykakasi
    _HAS_KAKASI = True
    _kks = pykakasi.kakasi()  # single analyzer; modes are implicit in convert()

    def jp_with_readings(text: str):
        tokens = _kks.convert(text)  # list of dicts with keys like: orig, hira, kana, hepburn, kunrei, passport
        hira = " ".join(t.get("hira", "") for t in tokens).strip()
        roma = " ".join(t.get("hepburn", "") for t in tokens).strip()  # choose 'kunrei' or 'passport' if preferred
        return hira, roma
except Exception:
    _HAS_KAKASI = False
    def jp_with_readings(_): return "", ""

# --- Whisper, Pyannote
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment

# --- torch/torchaudio for preload fallback + device detection
import torch
import torchaudio

# -----------------------
# Normalization / cleanup & atomic write helpers
# -----------------------

def _normalize_text_nfc(s: str) -> str:
    """Normalize to NFC and collapse repeated whitespace, trim edges."""
    if not isinstance(s, str):
        return s
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _atomic_write_text(path: Path, content: str, encoding: str = "utf-8"):
    """Write text atomically to avoid partial files on crashes."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding=encoding, newline="") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)

# -----------------------
# Utilities
# -----------------------

# ---- version helper (robust to packages without __version__)
def _pkg_version(dist_name: str, import_name: str | None = None):
    """
    Prefer importlib.metadata version by distribution name (pip name),
    fall back to importing the module and reading __version__ if present.
    """
    try:
        # Python 3.8+: importlib.metadata is in stdlib
        from importlib.metadata import version as _dist_version
        return _dist_version(dist_name)
    except Exception:
        pass
    try:
        mod = __import__(import_name or dist_name, fromlist=['__version__'])
        return getattr(mod, "__version__", None)
    except Exception:
        return None

def hms(seconds: float) -> str:
    ms = int(round(seconds * 1000.0))
    hrs = ms // 3_600_000
    ms %= 3_600_000
    mins = ms // 60_000
    ms %= 60_000
    secs = ms // 1000
    ms %= 1000
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"

def vtt_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000.0))
    hrs = ms // 3_600_000
    ms %= 3_600_000
    mins = ms // 60_000
    ms %= 60_000
    secs = ms // 1000
    ms %= 1000
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{ms:03d}"

def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    left = max(a_start, b_start)
    right = min(a_end, b_end)
    return max(0.0, right - left)

def relabel_speakers(speaker_list: List[str]) -> Dict[str, str]:
    mapping = {}
    counter = 1
    for s in speaker_list:
        if s not in mapping:
            mapping[s] = f"Speaker {counter}"
            counter += 1
    return mapping

def looks_japanese(s: str) -> bool:
    """Heuristic: contains any Hiragana/Katakana/Kanji codepoints."""
    for ch in s:
        code = ord(ch)
        if (0x3040 <= code <= 0x309F) or (0x30A0 <= code <= 0x30FF) or (0x4E00 <= code <= 0x9FFF):
            return True
    return False

def load_audio_ta(path: str, target_sr: int = 16000) -> dict:
    """Preload audio with torchaudio (mono, float32, resampled), for pyannote fallback."""
    wav, sr = torchaudio.load(path)  # (channels, time)
    if wav.dtype != torch.float32:
        wav = wav.float()
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # mono
    return {"waveform": wav, "sample_rate": sr}

def load_audio_ffmpeg(path: str, target_sr: int = 16000) -> dict:
    """
    Decode with FFmpeg CLI to mono float32 PCM at target_sr.
    Returns a dict compatible with pyannote's preloaded-audio path.
    """
    cmd = [
        "ffmpeg","-nostdin","-threads","0","-i",path,
        "-vn","-ac","1","-ar",str(target_sr),
        "-f","f32le","pipe:1"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed (code {proc.returncode}): {proc.stderr.decode(errors='ignore')[:500]}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise RuntimeError("ffmpeg produced no audio samples.")
    wav = torch.from_numpy(audio).unsqueeze(0)  # (1, T) mono
    return {"waveform": wav, "sample_rate": target_sr}

def probe_duration_ffprobe(path: str) -> Optional[float]:
    """
    Probe media duration using ffprobe. Returns seconds (float) or None.
    """
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1",
            path,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0:
            return None
        out = proc.stdout.decode("utf-8", errors="ignore").strip()
        return float(out) if out else None
    except Exception:
        return None


def ffmpeg_denoise(input_path: str, model_path: str, target_sr: int = 16000) -> str:
    """
    Denoise with RNNoise (arnndn) and resample to target_sr mono.
    Returns path to a temporary cleaned WAV.
    """
    tmp = tempfile.NamedTemporaryFile(prefix="clean_", suffix=".wav", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg","-y","-nostdin","-threads","0","-i",input_path,
        "-vn","-ac","1","-ar",str(target_sr),
        "-af", f"arnndn=m={model_path}",
        tmp.name
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        try: os.unlink(tmp.name)
        except Exception: pass
        raise RuntimeError(f"ffmpeg denoise failed (code {proc.returncode}): {proc.stderr.decode(errors='ignore')[:500]}")
    return tmp.name

# -----------------------
# Data classes
# -----------------------

@dataclass
class Turn:
    start: float
    end: float
    speaker: str
    text: str
    language: str
    lang_score: float

# -----------------------
# Core logic
# -----------------------

def transcribe_audio(
    audio_path: str,
    model_size: str = "large-v3",
    compute_type: str = "float16",
    beam_size: int = 5,
    vad_filter: bool = True,
    word_timestamps: bool = True,
    condition_on_previous_text: bool = True,
    language: Optional[str] = None,
    patience: Optional[float] = None,
    compression_ratio_threshold: Optional[float] = None,
    no_speech_threshold: Optional[float] = None,
    initial_prompt: Optional[str] = None,
    verbose: bool = False,
    progress: bool = False,
    total_duration: Optional[float] = None,
) -> List[Dict]:
    """
    Run faster-whisper transcription.

    - beam_size: wider search improves accuracy, slower decode.
    - word_timestamps: per-word timing (slower, more detail).
    - condition_on_previous_text: improves cross-segment consistency.
    - language: lock language code if known (e.g., "en", "ja"); None = auto.
    - patience, compression_ratio_threshold, no_speech_threshold: advanced controls.
    - initial_prompt: prime the decoder with domain-specific terms/names.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        logging.info(f"[Whisper] device={device} model={model_size} compute_type={compute_type} beam={beam_size} vad={vad_filter}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    kwargs = dict(
        task="transcribe",
        language=language if language else None,
        beam_size=beam_size,
        vad_filter=vad_filter,
        word_timestamps=word_timestamps,
        condition_on_previous_text=condition_on_previous_text,
        initial_prompt=initial_prompt,
    )
    if patience is not None: kwargs["patience"] = patience
    if compression_ratio_threshold is not None: kwargs["compression_ratio_threshold"] = compression_ratio_threshold
    if no_speech_threshold is not None: kwargs["no_speech_threshold"] = no_speech_threshold

    segments = []
    # time-based progress bar using known media duration if available
    pbar = None
    last_prog = 0.0
    try:
        if progress and total_duration and total_duration > 0:
            pbar = tqdm(total=total_duration, desc="Transcribing", unit="s", leave=True)
        # Single transcription call driving progress
        segments_gen, info = model.transcribe(audio_path, **kwargs)
        for seg in segments_gen:
            segments.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
            if pbar is not None:
                cur = max(0.0, float(seg.end))
                delta = max(0.0, cur - last_prog)
                if delta:
                    pbar.update(delta)
                    last_prog = cur
    finally:
        if pbar is not None:
            # Clamp to total if we overshot slightly and close
            try:
                remaining = max(0.0, pbar.total - pbar.n)
                if remaining:
                    pbar.update(remaining)
            except Exception:
                pass
            pbar.close()

    if verbose:
        logging.info(f"[Whisper] primary language: {getattr(info, 'language', None)} "
                     f"(p={getattr(info, 'language_probability', None)}) | segments={len(segments)}")
    return segments

def diarize_speakers(
    audio_input: Union[str, dict],
    hf_token: str,
    pipeline_name: str = "pyannote/speaker-diarization-3.1",
    num_speakers: Optional[int] = None,
    verbose: bool = False,
) -> Annotation:
    """
    Diarize speakers using Pyannote.
    Compatible with both the new DiarizeOutput (v3.1+) and older Annotation formats.
    Automatically tries FFmpeg CLI first, then torchaudio fallback.
    """
    # NOTE: main() guards calls to this when token is missing or diarization is skipped.
    if not hf_token:
        raise RuntimeError("Hugging Face token not found. This function expects a token.")

    if verbose:
        logging.info(f"[Pyannote] loading pipeline: {pipeline_name}")

    # Parse model id / revision
    model_id = pipeline_name
    revision = None
    if "@" in model_id:
        model_id, revision = model_id.split("@", 1)
        if verbose:
            logging.info(f"[Pyannote] using model_id='{model_id}' revision='{revision}'")

    pipeline = Pipeline.from_pretrained(model_id, revision=revision, token=hf_token)
    device = "cuda" if torch and torch.cuda.is_available() else "cpu"
    pipeline.to(torch.device(device))

    try_ffmpeg_first = isinstance(audio_input, str)
    if verbose:
        logging.info(f"[Pyannote] ffmpeg_cli_available=True | input_is_path={isinstance(audio_input, str)}")

    def _extract_annotation(result):
        """Handle both old (Annotation) and new (DiarizeOutput) pipeline results."""
        if hasattr(result, "annotation"):
            return result.annotation
        return result

    # 1) Try FFmpeg CLI path first
    if try_ffmpeg_first:
        try:
            if verbose:
                logging.info("[Pyannote] trying FFmpeg CLI decode path")
            audio_dict = load_audio_ffmpeg(audio_input)  # mono float32 16k
            result = pipeline(audio_dict, num_speakers=num_speakers)
            return _extract_annotation(result)
        except Exception as e:
            if verbose:
                logging.warning(f"[Pyannote] FFmpeg path failed ({e.__class__.__name__}: {e}); falling back to torchaudio preload")

    # 2) Fallback to preloaded torchaudio
    if isinstance(audio_input, str):
        audio_dict = load_audio_ta(audio_input)
    else:
        audio_dict = audio_input

    if verbose:
        logging.info("[Pyannote] using preloaded-audio (torchaudio) path")
    result = pipeline(audio_dict, num_speakers=num_speakers)
    return _extract_annotation(result)

def assign_speakers_to_asr(asr: List[Dict], diarization) -> List[Tuple[Dict, str]]:
    """
    Map speakers from pyannote output to ASR segments.
    Accepts:
      - pyannote.core.Annotation
      - wrapper with .annotation (Annotation)
      - wrapper with .speaker_diarization iterable of (Segment, speaker) pairs
    """
    # Case 1: already an Annotation
    if hasattr(diarization, "itertracks"):
        ann = diarization
    # Case 2: wrapper exposing .annotation -> Annotation
    elif hasattr(diarization, "annotation"):
        ann = diarization.annotation
        if not hasattr(ann, "itertracks"):
            raise TypeError("Unwrapped .annotation is not an Annotation-like object.")
    # Case 3: wrapper exposing .speaker_diarization -> build an Annotation
    elif hasattr(diarization, "speaker_diarization"):
        ann = Annotation()
        for turn, speaker in diarization.speaker_diarization:
            ann[turn] = str(speaker)
    else:
        raise TypeError(
            f"Unexpected diarization type: {type(diarization)}. "
            "Expected Annotation, or an object with .annotation or .speaker_diarization."
        )

    turns = list(ann.itertracks(yield_label=True))  # (Segment, track, label)
    assigned: List[Tuple[Dict, str]] = []
    for seg in asr:
        s_start, s_end = seg["start"], seg["end"]
        best_label, best_olap = "Unknown", 0.0
        for (turn, _track, label) in turns:
            ol = overlap(s_start, s_end, turn.start, turn.end)
            if ol > best_olap:
                best_olap = ol
                best_label = label
        assigned.append((seg, best_label))
    return assigned

def detect_language_for_text(text: str) -> Tuple[str, float]:
    if not text.strip():
        return "und", 0.0
    code, score = langid.classify(text)
    return code, float(score)

def build_turns(asr_segments: List[Dict], diarization: Annotation, show_progress: bool = False) -> List[Turn]:
    speaker_assigned = assign_speakers_to_asr(asr_segments, diarization)
    turns: List[Turn] = []
    it = tqdm(speaker_assigned, desc="Tagging language / merging", disable=not show_progress)
    for seg, spk in it:
        # Clean punctuation/spacing and normalize to NFC before downstream use
        cleaned_text = _normalize_text_nfc(seg["text"])
        code, score = detect_language_for_text(cleaned_text)
        turns.append(Turn(
            start=float(seg["start"]),
            end=float(seg["end"]),
            speaker=spk,
            text=cleaned_text,
            language=code,
            lang_score=score,
        ))
    turns.sort(key=lambda t: t.start)
    mapping = relabel_speakers([t.speaker for t in turns])
    for t in turns:
        t.speaker = mapping.get(t.speaker, t.speaker)
    return turns

def merge_adjacent(turns: List[Turn], max_gap: float = 0.3) -> List[Turn]:
    """Merge consecutive turns by same speaker when gap <= max_gap seconds."""
    if not turns:
        return turns
    merged: List[Turn] = []
    for t in sorted(turns, key=lambda x: x.start):
        if merged and merged[-1].speaker == t.speaker and (t.start - merged[-1].end) <= max_gap:
            merged[-1].end = max(merged[-1].end, t.end)
            merged[-1].text = _normalize_text_nfc((merged[-1].text + " " + t.text).strip())
        else:
            merged.append(t)
    return merged

# -----------------------
# Writers + headers
# -----------------------

def _format_settings_header(meta: Dict[str, Union[str,int,float,bool,None]]) -> str:
    pairs = []
    for k, v in meta.items():
        if isinstance(v, (list, dict)):  # keep it simple and readable
            v = json.dumps(v, ensure_ascii=False)
        pairs.append(f"{k}: {v}")
    return "\n".join(pairs)

def write_json(turns: List[Turn], path: Path):
    # Preserve original schema (list of turns) for compatibility
    data = [asdict(t) for t in turns]
    content = json.dumps(data, ensure_ascii=False, indent=2)
    _atomic_write_text(path, content)

def write_meta_json(meta: Dict[str, Union[str,int,float,bool,None]], path: Path):
    content = json.dumps(meta, ensure_ascii=False, indent=2)
    _atomic_write_text(path, content)

def _format_ts_for_example(seconds: float) -> str:
    # hh:mm:ss (no milliseconds) for compact examples
    ms = int(round(seconds * 1000.0))
    hrs = ms // 3_600_000
    ms %= 3_600_000
    mins = ms // 60_000
    ms %= 60_000
    secs = ms // 1000
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

def _looks_japanese_text(s: str) -> bool:
    # mirrors looks_japanese but works on small tokens too
    for ch in s:
        code = ord(ch)
        if (0x3040 <= code <= 0x309F) or (0x30A0 <= code <= 0x30FF) or (0x4E00 <= code <= 0x9FFF):
            return True
    return False

def build_glossary(turns: List[Turn]) -> List[Dict[str, str]]:
    """
    Collect unique Japanese terms from JA segments (or lines that look Japanese),
    using pykakasi tokenization to get readings. Count frequencies and keep first example time.
    """
    entries: Dict[str, Dict[str, Union[str, int, float]]] = {}
    for t in turns:
        if not (t.language == "ja" or _looks_japanese_text(t.text)):
            continue
        text = t.text.strip()
        if not text:
            continue
        # Tokenize via pykakasi if available; else very rough fallback by Kanji/Hiragana/Katakana runs
        tokens = []
        if _HAS_KAKASI:
            for token in _kks.convert(text):
                orig = (token.get("orig") or "").strip()
                if not orig or not _looks_japanese_text(orig):
                    continue
                hira = (token.get("hira") or "").strip()
                roma = (token.get("hepburn") or "").strip()
                tokens.append((orig, hira, roma))
        else:
            # Fallback: split by non-CJK and keep CJK runs as "terms"
            for m in re.finditer(r"[\u3040-\u30FF\u4E00-\u9FFF]+", text):
                term = m.group(0)
                tokens.append((term, "", ""))

        for orig, hira, roma in tokens:
            if not orig:
                continue
            if orig not in entries:
                entries[orig] = {
                    "term": orig,
                    "hira": hira,
                    "roma": roma,
                    "count": 0,
                    "first_ts": _format_ts_for_example(t.start),
                    "example": text,
                }
            entries[orig]["count"] += 1  # type: ignore

    # Optional Spanish translation (only if Argos present and study_translate was requested)
    if globals().get("_HAS_ARGOS") and globals().get("_STUDY_TRANSLATE", False):
        for k, e in entries.items():
            term = e["term"]  # type: ignore
            try:
                es = translate_ja_to_es(term) or ""
            except Exception:
                es = ""
            e["es"] = es  # type: ignore

    # Sort by frequency desc, then term
    sorted_entries = sorted(entries.values(), key=lambda d: (-int(d["count"]), str(d["term"])))  # type: ignore
    return [  # cast to list of str->str for writing
        {
            "term": str(e.get("term", "")),
            "hira": str(e.get("hira", "")),
            "roma": str(e.get("roma", "")),
            "count": str(e.get("count", "")),
            "first_ts": str(e.get("first_ts", "")),
            "example": str(e.get("example", "")),
            "es": str(e.get("es", "")) if "es" in e else "",
        } for e in sorted_entries
    ]

def write_glossary(gloss: List[Dict[str, str]], path: Path, settings_header: str):
    """
    Write a human-friendly glossary. One entry per block:
      term | hira | roma | count | first_ts
      es? (if present)
      example: ...
    """
    buf = io.StringIO()
    buf.write("# Glossary (auto-collected from Japanese segments)\n")
    for line in settings_header.splitlines():
        buf.write("# " + line + "\n")
    buf.write("\n")
    if not gloss:
        buf.write("(No Japanese terms detected.)\n")
        _atomic_write_text(path, buf.getvalue())
        return
    for e in gloss:
        # Normalize key fields
        e = {k: _normalize_text_nfc(v or "") for k, v in e.items()}
        header = f"{e['term']}"
        extras = []
        if e.get("hira"): extras.append(f"hira: {e['hira']}")
        if e.get("roma"): extras.append(f"romaji: {e['roma']}")
        if e.get("count"): extras.append(f"count: {e['count']}")
        if e.get("first_ts"): extras.append(f"first: {e['first_ts']}")
        buf.write(header + ("  |  " + "  |  ".join(extras) if extras else "") + "\n")
        if e.get("es"):
            buf.write(f"  es: {e['es']}\n")
        if e.get("example"):
            buf.write(f"  example: {e['example']}\n")
        buf.write("\n")
    _atomic_write_text(path, buf.getvalue())

def write_srt(turns: List[Turn], path: Path, settings_header: str):
    # SRT has no official comment; add a zero-length cue #0 with settings
    buf = io.StringIO()
    buf.write("0\n00:00:00,000 --> 00:00:00,001\n")
    buf.write("[SETTINGS]\n" + settings_header + "\n\n")
    for i, t in enumerate(turns, start=1):
        spk = _normalize_text_nfc(t.speaker)
        txt = _normalize_text_nfc(t.text)
        lang_tag = t.language if t.language else "und"
        buf.write(f"{i}\n")
        buf.write(f"{hms(t.start)} --> {hms(t.end)}\n")
        buf.write(f"{spk} [{lang_tag}]: {txt}\n\n")
    _atomic_write_text(path, buf.getvalue())

def write_vtt(turns: List[Turn], path: Path, settings_header: str):
    buf = io.StringIO()
    buf.write("WEBVTT\n\n")
    buf.write("NOTE\n")
    buf.write("[SETTINGS]\n" + settings_header + "\n\n")
    for t in turns:
        spk = _normalize_text_nfc(t.speaker)
        txt = _normalize_text_nfc(t.text)
        lang_tag = t.language if t.language else "und"
        buf.write(f"{vtt_ts(t.start)} --> {vtt_ts(t.end)}\n")
        buf.write(f"{spk} [{lang_tag}]: {txt}\n\n")
    _atomic_write_text(path, buf.getvalue())

def write_txt(turns: List[Turn], path: Path, settings_header: str):
    buf = io.StringIO()
    buf.write("# Transcript\n")
    buf.write("# ---------\n")
    for line in settings_header.splitlines():
        buf.write("# " + line + "\n")
    buf.write("\n")
    for t in turns:
        spk = _normalize_text_nfc(t.speaker)
        txt = _normalize_text_nfc(t.text)
        buf.write(f"[{hms(t.start)[:-4]}] {spk} ({t.language}): {txt}\n")
    _atomic_write_text(path, buf.getvalue())

def write_csv(turns: List[Turn], path: Path, settings_header: str):
    # Prepend commented header lines, then write CSV with robust quoting
    # Build entire content in-memory to ensure atomic replace after success
    header = io.StringIO()
    header.write("# SETTINGS\n")
    for line in settings_header.splitlines():
        header.write("# " + line + "\n")
    header.write("# ---\n")
    csv_buf = io.StringIO(newline="")
    writer = csv.writer(csv_buf, quoting=csv.QUOTE_ALL)
    writer.writerow(["start", "end", "speaker", "language", "lang_score", "text"])
    for t in turns:
        writer.writerow([
            t.start,
            t.end,
            _normalize_text_nfc(t.speaker),
            t.language,
            t.lang_score,
            _normalize_text_nfc(t.text).replace("\n", " ")
        ])
    _atomic_write_text(path, header.getvalue() + csv_buf.getvalue())

def write_study(turns: List[Turn], path: Path, settings_header: str, include_hiragana: bool = True, include_romaji: bool = True):
    buf = io.StringIO()
    buf.write("# Study Transcript (speaker order, with Japanese readings)\n")
    for line in settings_header.splitlines():
        buf.write("# " + line + "\n")
    buf.write("\n")
    if not _HAS_KAKASI and any((t.language == "ja" or looks_japanese(t.text)) for t in turns):
        buf.write("(Note: pykakasi not installed — hiragana/romaji hints disabled)\n\n")
    for t in turns:
        spk = _normalize_text_nfc(t.speaker)
        txt = _normalize_text_nfc(t.text)
        buf.write(f"[{hms(t.start)[:-4]}] {spk} ({t.language})\n")
        buf.write(txt + "\n")
        if (t.language == "ja" or looks_japanese(t.text)) and _HAS_KAKASI:
            hira, roma = jp_with_readings(txt)
            if include_hiragana and hira.strip():
                buf.write(f"〔hiragana〕 {hira}\n")
            if include_romaji and roma.strip():
                buf.write(f"〔romaji〕 {roma}\n")
        if (t.language == "ja" or looks_japanese(t.text)) and globals().get("_HAS_ARGOS") and globals().get("_STUDY_TRANSLATE", False):
            es = translate_ja_to_es(txt)
            if es:
                buf.write(f"〔español〕 {es}\n")
        buf.write("\n")
    _atomic_write_text(path, buf.getvalue())

# -----------------------
# Offline translation (Argos Translate)
# -----------------------
try:
    import argostranslate.package, argostranslate.translate
    _HAS_ARGOS = True

    def _ensure_argos_models():
        """Ensure the JA→EN and EN→ES models are installed (idempotent)."""
        try:
            argostranslate.package.update_package_index()
            available = argostranslate.package.get_available_packages()
            installed = {(p.from_code, p.to_code) for p in argostranslate.package.get_installed_packages()}

            def _install(from_code: str, to_code: str):
                if (from_code, to_code) in installed:
                    return
                pkg = next((p for p in available if p.from_code == from_code and p.to_code == to_code), None)
                if pkg is None:
                    logging.warning(f"[Argos] No package found for {from_code}→{to_code}")
                    return
                logging.info(f"[Argos] Installing {from_code}→{to_code} model…")
                argostranslate.package.install_from_path(pkg.download())

            _install("ja", "en")
            _install("en", "es")
        except Exception as e:
            logging.warning(f"[Argos] Model ensure failed: {e}")

    def translate_ja_to_es(text: str) -> str:
        """Offline Japanese → English → Spanish using Argos Translate."""
        if not text or not text.strip():
            return ""
        try:
            _ensure_argos_models()
            en = argostranslate.translate.translate(text, "ja", "en")
            es = argostranslate.translate.translate(en, "en", "es")
            return es.strip()
        except Exception as e:
            logging.warning(f"[Argos] translate failed: {e}")
            return ""
except Exception:
    _HAS_ARGOS = False
    def translate_ja_to_es(_): return ""

# -----------------------
# CLI
# -----------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Diarize speakers + transcribe + per-segment language ID using Whisper & Pyannote. "
                    "Optionally denoise with FFmpeg/RNNoise and emit study-friendly outputs."
    )
    ap.add_argument("input", 
                    help="Path to an audio/video file (anything FFmpeg can read: wav, mp3, mp4, mkv, mov, etc.).")
    ap.add_argument("--formats", default="json,srt,txt", 
                    help="Comma-separated outputs to generate: json,srt,vtt,txt,csv,study (default: json,srt,txt).")
    ap.add_argument("--suffix", default="_whisper_diarized", 
                    help="Suffix added to output filenames (before extension), e.g., input_whisper_diarized.srt")
    # Whisper controls
    ap.add_argument("--model-size", default="large-v3",
                    help="Whisper model size: tiny/base/small/medium/large-v3 (default: large-v3). Larger is slower, more accurate.")
    ap.add_argument("--compute-type", default="float16",
                    help="Whisper compute type on GPU: float16,int8_float16,int8,float32 (default: float16). On CPU, prefer int8.")
    ap.add_argument("--beam-size", type=int, default=5,
                    help="Beam search width (higher = more accurate, slower). Try 1 (fast) to 8 (accurate). Default: 5.")
    ap.add_argument("--language", default="",
                    help="If known, lock language (e.g., 'en', 'ja', 'es'). Leave empty for auto-detect.")
    ap.add_argument("--patience", type=float, default=None,
                    help="Beam patience (e.g., 1.0–2.0). Allows slightly longer candidates during beam search.")
    ap.add_argument("--compression-ratio-threshold", type=float, default=None,
                    help="Stricter threshold reduces hallucinations (lower is stricter).")
    ap.add_argument("--no-speech-threshold", type=float, default=None,
                    help="Probability threshold to treat regions as silence (higher = more aggressive).")
    ap.add_argument("--initial-prompt", default=None,
                    help="Domain priming text (names/terms) to bias decoding at the start.")
    ap.add_argument("--no-vad", action="store_true",
                    help="Disable VAD filter (default is ON). Use only if words are being clipped.")
    # Decoding detail (default ON; flags disable)
    ap.add_argument("--no-word-timestamps", action="store_true",
                    help="Disable per-word timestamps (defaults to ON). Speeds up but reduces timing detail.")
    ap.add_argument("--no-condition-on-previous-text", action="store_true",
                    help="Disable conditioning on previous text (defaults to ON). Can reduce consistency.")
    # Denoise / preprocessing
    ap.add_argument("--ffmpeg-denoise-model", default=None,
                    help="Optional RNNoise .rnnn model path for FFmpeg arnndn denoise. If set, a cleaned temp WAV is generated and used.")
    # Pyannote controls
    ap.add_argument("--hf-token", default=os.environ.get("HUGGINGFACE_TOKEN", ""),
                    help="Hugging Face token for pyannote models (required).")
    ap.add_argument("--diarization-pipeline", default="pyannote/speaker-diarization-3.1",
                    help="Pyannote pipeline repo id (default: pyannote/speaker-diarization-3.1). Accepts 'repo@revision'.")
    ap.add_argument("--diarization-revision", default=None,
                    help="Optional explicit revision for the diarization pipeline (overrides any '@rev' in --diarization-pipeline).")
    ap.add_argument("--hf-cache-dir", default=None,
                    help="Optional Hugging Face cache directory to reduce repeated downloads (sets HF_HOME for this run).")
    ap.add_argument("--num-speakers", type=int, default=None,
                    help="If known, fix the number of speakers (improves diarization).")
    ap.add_argument("--preload-audio", action="store_true",
                    help="Force preloaded-audio (torchaudio) path for diarization (bypass FFmpeg).")
    ap.add_argument("--skip-diarization", action="store_true",
                    help="Skip speaker diarization (treat as a single speaker). Useful when no HF token is available.")
    # Misc
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Verbose logging + progress bars.")
    ap.add_argument("--progress", action="store_true",
                    help="Show progress bars even when --verbose is not set.")
    # Study options
    ap.add_argument("--study-no-hiragana", action="store_true", 
                    help="Omit hiragana line in study output.")
    ap.add_argument("--study-no-romaji", action="store_true", 
                    help="Omit romaji line in study output.")
    ap.add_argument("--no-study-translate", action="store_true",
                    help="Disable Spanish translation for Japanese segments in the study output.")
    # Glossary
    ap.add_argument("--glossary", action="store_true",
                    help="Emit a Japanese term glossary alongside other outputs (suffix: .glossary.txt).")
    ap.add_argument("--merge-gap", type=float, default=0.3,
                    help="Seconds gap to merge consecutive turns by the same speaker (default: 0.3).")
    return ap.parse_args()

# -----------------------
# Main
# -----------------------

def main():
    args = parse_args()
    _t0_total = time.perf_counter()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Extra silencing when not verbose: kill all INFO/DEBUG emitted by 3rd parties
    if not args.verbose:
        # Disable INFO and below globally (affects root and all child loggers)
        logging.disable(logging.INFO)
        # Additionally clamp a few known noisy libraries
        for name, lvl in {
            "argostranslate": logging.ERROR,
            "ctranslate2": logging.WARNING,
            "faster_whisper": logging.WARNING,
            "pyannote.audio": logging.WARNING,
            "transformers": logging.ERROR,
            "urllib3": logging.ERROR,
            "httpx": logging.ERROR,
            "numba": logging.ERROR,
        }.items():
            logging.getLogger(name).setLevel(lvl)
            logging.getLogger(name).propagate = False

    in_path = Path(args.input)
    if not in_path.exists():
        logging.error(f"Input not found: {in_path}")
        sys.exit(1)

    # ---- Early FFmpeg/FFprobe guards (fail fast with clear hints)
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            print(
                f"ERROR: Required tool '{tool}' was not found on your PATH.\n"
                "Install hints:\n"
                "  - macOS:   brew install ffmpeg\n"
                "  - Ubuntu:  sudo apt-get update && sudo apt-get install -y ffmpeg\n"
                "  - Windows: winget install Gyan.FFmpeg (then restart shell)\n",
                file=sys.stderr
            )
            sys.exit(1)

    # Optional denoise/preprocess
    temp_path = None
    working_input = str(in_path)
    if args.ffmpeg_denoise_model:
        try:
            if args.verbose:
                logging.info("[Preprocess] Denoising with FFmpeg arnndn")
            pbar_dn = tqdm(total=1, desc="Denoising", disable=not (args.verbose or args.progress))
            temp_path = ffmpeg_denoise(str(in_path), args.ffmpeg_denoise_model, target_sr=16000)
            working_input = temp_path
            pbar_dn.update(1); pbar_dn.close()
        except Exception as e:
            try:
                pbar_dn.close()
            except Exception:
                pass
            logging.warning(f"[Preprocess] Denoise failed: {e}. Continuing with original input.")

    # Transcribe (Whisper) with timing
    _t0_asr = time.perf_counter()
    # Probe duration for a smoother progress bar
    media_duration = probe_duration_ffprobe(working_input)
    asr_segments = transcribe_audio(
        working_input,
        model_size=args.model_size,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        vad_filter=not args.no_vad,
        word_timestamps=not args.no_word_timestamps,
        condition_on_previous_text=not args.no_condition_on_previous_text,
        language=args.language or None,
        patience=args.patience,
        compression_ratio_threshold=args.compression_ratio_threshold,
        no_speech_threshold=args.no_speech_threshold,
        initial_prompt=args.initial_prompt,
        verbose=args.verbose,
        progress=(args.verbose or args.progress),
        total_duration=media_duration,
    )
    asr_seconds = time.perf_counter() - _t0_asr

    # Basic text/tempo metrics from ASR
    total_words = sum(len(s["text"].split()) for s in asr_segments)
    # sum of segment durations (spoken time proxy)
    spoken_seconds = sum(max(0.0, (s["end"] - s["start"])) for s in asr_segments)
    # full audio duration proxy from ASR (max end); robust and avoids extra probes
    audio_seconds = max((s["end"] for s in asr_segments), default=0.0)
    wpm_full = (total_words / (audio_seconds / 60.0)) if audio_seconds > 0 else None
    wpm_spoken = (total_words / (spoken_seconds / 60.0)) if spoken_seconds > 0 else None

    # Decide diarization input
    if args.preload_audio:
        audio_input: Union[str, dict] = load_audio_ta(working_input)
        if args.verbose:
            logging.info("[Main] --preload-audio requested; bypassing FFmpeg CLI")
    else:
        audio_input = working_input

    # Honor HF cache dir to reduce repeated pulls
    if args.hf_cache_dir:
        os.environ["HF_HOME"] = args.hf_cache_dir

    # Diarize (Pyannote) with timing — allow graceful skip when no token or --skip-diarization, and fallback on errors
    _t0_diar = time.perf_counter()
    diar_seconds = 0.0
    diar = None
    do_diarization = (not args.skip_diarization) and bool(args.hf_token)
    if do_diarization:
        pbar_dz = None
        try:
            if (args.verbose or args.progress):
                pbar_dz = tqdm(total=1, desc="Diarizing", disable=False)
            # Choose revision: explicit CLI overrides repo@rev parsing
            pipeline_name = args.diarization_pipeline
            if args.diarization_revision:
                if "@" in pipeline_name:
                    pipeline_name = pipeline_name.split("@", 1)[0]
                pipeline_name = f"{pipeline_name}@{args.diarization_revision}"
            diar = diarize_speakers(
                audio_input=audio_input,
                hf_token=args.hf_token,
                pipeline_name=pipeline_name,
                num_speakers=args.num_speakers,
                verbose=args.verbose
            )
        except Exception as e:
            logging.warning(f"[Main] Diarization failed ({e.__class__.__name__}: {e}); falling back to single-speaker.")
            diar = None
        finally:
            if pbar_dz is not None:
                pbar_dz.update(1)
                pbar_dz.close()
        diar_seconds = time.perf_counter() - _t0_diar
    else:
        if not args.hf_token and not args.skip_diarization:
            logging.warning("[Main] --hf-token not provided; skipping diarization (single-speaker fallback).")
    # Build a simple single-speaker annotation covering the whole audio when skipped or failed
    if diar is None:
        diar = Annotation()
        diar[Segment(0.0, max((s['end'] for s in asr_segments), default=0.0))] = "Speaker 1"

    # Unwrap DiarizeOutput if needed
    if hasattr(diar, "annotation"):
        diar = diar.annotation

    # Merge + per-segment language ID
    turns = build_turns(asr_segments, diar, show_progress=(args.verbose or args.progress))
    # Coalesce small gaps between same-speaker turns
    turns = merge_adjacent(turns, max_gap=float(args.merge_gap))

    # Per-speaker metrics (words and speaking time)
    speaker_stats = defaultdict(lambda: {"words": 0, "speaking_seconds": 0.0})
    for t in turns:
        speaker_stats[t.speaker]["words"] += len(t.text.split())
        speaker_stats[t.speaker]["speaking_seconds"] += max(0.0, t.end - t.start)
    speaker_metrics = {
        spk: {
            "words": st["words"],
            "speaking_seconds": round(st["speaking_seconds"], 3),
            "wpm_spoken": (st["words"] / (st["speaking_seconds"] / 60.0)) if st["speaking_seconds"] > 0 else None,
        }
        for spk, st in speaker_stats.items()
    }

    # Prepare settings meta
    device = "cuda" if torch.cuda.is_available() else "cpu"

    total_seconds = time.perf_counter() - _t0_total

    meta = {
        "input": str(in_path),
        "used_input": working_input,
        "device": device,
        "whisper_model": args.model_size,
        "compute_type": args.compute_type,
        "beam_size": args.beam_size,
        "word_timestamps": not args.no_word_timestamps,
        "condition_on_previous_text": not args.no_condition_on_previous_text,
        "language": args.language or None,
        "patience": args.patience,
        "compression_ratio_threshold": args.compression_ratio_threshold,
        "no_speech_threshold": args.no_speech_threshold,
        "initial_prompt": bool(args.initial_prompt),
        "vad_filter": not args.no_vad,
        "ffmpeg_denoise_model": args.ffmpeg_denoise_model or None,
        "diarization_pipeline": args.diarization_pipeline,
        "diarization_revision": args.diarization_revision,
        "hf_cache_dir": args.hf_cache_dir,
        "num_speakers": args.num_speakers,
        "skip_diarization": not do_diarization,
        "pyannote_device": device,
        "study_translate": not args.no_study_translate,
        "merge_gap": float(args.merge_gap),
        "metrics": {
            "transcribe_seconds": round(asr_seconds, 3),
            "diarization_seconds": round(diar_seconds, 3),
            "total_seconds": round(total_seconds, 3),
            "audio_seconds_estimate": round(audio_seconds, 3),
            "spoken_seconds_sum": round(spoken_seconds, 3),
            "total_words": int(total_words),
            "wpm_over_full_audio": (round(wpm_full, 2) if wpm_full is not None else None),
            "wpm_over_spoken_time": (round(wpm_spoken, 2) if wpm_spoken is not None else None),
            "per_speaker": speaker_metrics,
        },
        "versions": {
            "python": sys.version.split()[0],
            "torch": getattr(torch, "__version__", None),
            "torchaudio": getattr(torchaudio, "__version__", None),
            "faster-whisper": _pkg_version("faster-whisper", "faster_whisper"),
            "pyannote.audio": _pkg_version("pyannote.audio", "pyannote.audio"),
            "langid": _pkg_version("langid", "langid"),
        }
    }
    settings_header = _format_settings_header(meta)

    # Outputs — same directory as input
    base = in_path.with_suffix("")  # drop extension
    fmts = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    out_paths: Dict[str, Path] = {}

    # Make study translation flag visible to writer without changing its signature
    globals()["_STUDY_TRANSLATE"] = not args.no_study_translate

    # Emit machine-readable meta **only** when JSON is requested
    if "json" in fmts:
        meta_path = Path(f"{base}{args.suffix}.meta.json")
        write_meta_json(meta, meta_path)
        out_paths["meta"] = meta_path
        p = Path(f"{base}{args.suffix}.json")
        write_json(turns, p)
        out_paths["json"] = p
    if "srt" in fmts:
        p = Path(f"{base}{args.suffix}.srt"); write_srt(turns, p, settings_header); out_paths["srt"] = p
    if "vtt" in fmts:
        p = Path(f"{base}{args.suffix}.vtt"); write_vtt(turns, p, settings_header); out_paths["vtt"] = p
    if "txt" in fmts:
        p = Path(f"{base}{args.suffix}.txt"); write_txt(turns, p, settings_header); out_paths["txt"] = p
    if "csv" in fmts:
        p = Path(f"{base}{args.suffix}.csv"); write_csv(turns, p, settings_header); out_paths["csv"] = p
    if "study" in fmts:
        p = Path(f"{base}{args.suffix}.study.txt")
        write_study(turns, p, settings_header,
                    include_hiragana=not args.study_no_hiragana,
                    include_romaji=not args.study_no_romaji)
        out_paths["study"] = p
    if args.glossary:
        g = build_glossary(turns)
        p = Path(f"{base}{args.suffix}.glossary.txt")
        write_glossary(g, p, settings_header)
        out_paths["glossary"] = p

    if args.verbose:
        for k, p in out_paths.items():
            logging.info(f"Wrote {k.upper()}: {p}")

    print("Done. Outputs:")
    for k, p in out_paths.items():
        print(f"  - {k.upper()}: {p}")

    # ---- Console timing summary
    try:
        print("\nSummary:")
        print(f"  Transcription time : {asr_seconds:.2f}s")
        print(f"  Diarization time   : {diar_seconds:.2f}s")
        print(f"  Total runtime      : {total_seconds:.2f}s")
        if total_words:
            print(f"  Words              : {int(total_words)}")
        if wpm_full is not None:
            print(f"  WPM (full audio)   : {wpm_full:.2f}")
        if wpm_spoken is not None:
            print(f"  WPM (spoken time)  : {wpm_spoken:.2f}")
    except NameError:
        # metrics not available (older script) — skip gracefully
        pass

    # Cleanup temp
    if temp_path:
        try: os.unlink(temp_path)
        except Exception: pass

if __name__ == "__main__":
    main()

# (end)
