#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
conda py312


python  /mnt/c/Users/epena/Projects/OpenAI_Audio_Speech_Recognition/whisper_diarize_multilang2.py --formats json,srt,txt,study   --beam-size 8   --patience 1.5   --compression-ratio-threshold 2.0   --no-speech-threshold 0.6   --initial-prompt "Clase de japonés en español. Sensei, alumno, romaji, hiragana, katakana, Kanji, partícula, conjugación, ejemplo, pregunta, respuesta, vocabulario, です, ます, は, を, が."   --num-speakers 11   --hf-token <here-toke> --glossary   --no-condition-on-previous-text   --progress --n-experiments 3 --experiment-scheme accuracy --allowed-languages "ja,es" --redecode-offlang /mnt/c/Users/epena/Downloads/Japones_Oct_2025/20251121.mp4
"""


"""
Whisper + Pyannote diarization + per-segment language ID (langid)
+ Study-friendly TXT with kana/romaji for Japanese segments
+ Multi-run experiments via --n-experiments
+ JA/ES-only enforcement and off-language rescue
-----------------------------------------------------------------
- Inputs: an audio/video file path
- Outputs (per run): JSON / SRT / VTT / TXT / CSV / STUDY next to the input file
  using: <input><suffix>_expXX.<ext>
- Summary: one CSV with knobs & quality proxies: <input><suffix>_experiments.csv
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
import numpy as np
import time
import re
import io
import unicodedata
import shutil
import csv
from collections import defaultdict, Counter
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
        tokens = _kks.convert(text)  # list of dicts: orig, hira, kana, hepburn, kunrei, passport
        hira = " ".join(t.get("hira", "") for t in tokens).strip()
        roma = " ".join(t.get("hepburn", "") for t in tokens).strip()
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

def _katakana_to_hiragana(s: str) -> str:
    out = []
    for ch in s:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F3:         # Katakana
            out.append(chr(code - 0x60))      # -> Hiragana
        else:
            out.append(ch)
    return "".join(out)

def _contains_kanji(s: str) -> bool:
    return any(0x4E00 <= ord(ch) <= 0x9FFF for ch in s)

# Try Sudachi first
try:
    from sudachipy import dictionary as _sd_dict
    from sudachipy import tokenizer as _sd_tok
    _HAS_SUDACHI = True
    _SUDACHI = _sd_dict.Dictionary().create()
    _SUDACHI_MODE = _sd_tok.Tokenizer.SplitMode.C
except Exception:
    _HAS_SUDACHI = False

# Try fugashi second
try:
    import fugashi
    _HAS_FUGASHI = True
    _FUGASHI = fugashi.Tagger()
except Exception:
    _HAS_FUGASHI = False

# Filters
_JA_POLITE_AUX = {
    "です","ます","でした","でしたか","でしたね","ません","ませんでした",
    "たい","たいです","ください","ましょう","おります","おりますか","ございます",
}
_JA_DROP_EXACT = {
    # honorifics / suffixes / common function words
    "さん","ちゃん","様","さま","君","くん",
    "に","は","が","を","の","へ","で","と","や","か","も","な","ね","よ","ぞ","ぜ","さ",
    "から","まで","より","だけ","しか","でも","とは","など","って","たり","ので","のは",
    "これは","それでは","それを","それにも","この","その","どれだけ","どのくらい",
}
_ALLOWED_POS_TOP = {"名詞","動詞","形容詞","副詞"}
# Noun subcategories to drop (Sudachi)
_NOUN_SUB_DROP = {"数詞","助数詞","代名詞","接尾辞","非自立可能"}


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

def _pkg_version(dist_name: str, import_name: str | None = None):
    """
    Prefer importlib.metadata version by distribution name (pip name),
    fall back to importing the module and reading __version__ if present.
    """
    try:
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
    # NEW: allow passing a preloaded model for reuse
    _shared_model: Optional[WhisperModel] = None,
) -> Tuple[List[Dict], Dict]:
    """
    Run faster-whisper transcription and return (segments, info_dict).

    - language: lock language code if known (e.g., "en", "ja"); None = auto.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        logging.info(f"[Whisper] device={device} model={model_size} compute_type={compute_type} beam={beam_size} vad={vad_filter}")

    model = _shared_model or WhisperModel(model_size, device=device, compute_type=compute_type)

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
    info_fields = {
        "language": None,
        "language_probability": None,
        "duration": None,
    }
    pbar = None
    last_prog = 0.0
    try:
        if progress and total_duration and total_duration > 0:
            pbar = tqdm(total=total_duration, desc="Transcribing", unit="s", leave=True)
        segments_gen, info = model.transcribe(audio_path, **kwargs)
        # capture info if available
        for k in ("language", "language_probability", "duration"):
            if hasattr(info, k):
                info_fields[k] = getattr(info, k)
        for seg in segments_gen:
            segments.append({"start": seg.start, "end": seg.end, "text": (seg.text or "").strip()})
            if pbar is not None:
                cur = max(0.0, float(seg.end))
                delta = max(0.0, cur - last_prog)
                if delta:
                    pbar.update(delta)
                    last_prog = cur
    finally:
        if pbar is not None:
            try:
                remaining = max(0.0, pbar.total - pbar.n)
                if remaining:
                    pbar.update(remaining)
            except Exception:
                pass
            pbar.close()

    if verbose:
        logging.info(f"[Whisper] primary language: {info_fields.get('language')} "
                     f"(p={info_fields.get('language_probability')}) | segments={len(segments)}")
    return segments, info_fields

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
    if not hf_token:
        raise RuntimeError("Hugging Face token not found. This function expects a token.")

    if verbose:
        logging.info(f"[Pyannote] loading pipeline: {pipeline_name}")

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
        if hasattr(result, "annotation"):
            return result.annotation
        return result

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
    """
    if hasattr(diarization, "itertracks"):
        ann = diarization
    elif hasattr(diarization, "annotation"):
        ann = diarization.annotation
        if not hasattr(ann, "itertracks"):
            raise TypeError("Unwrapped .annotation is not an Annotation-like object.")
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
        if isinstance(v, (list, dict)):
            v = json.dumps(v, ensure_ascii=False)
        pairs.append(f"{k}: {v}")
    return "\n".join(pairs)

def write_json(turns: List[Turn], path: Path):
    data = [asdict(t) for t in turns]
    content = json.dumps(data, ensure_ascii=False, indent=2)
    _atomic_write_text(path, content)

def write_meta_json(meta: Dict[str, Union[str,int,float,bool,None]], path: Path):
    content = json.dumps(meta, ensure_ascii=False, indent=2)
    _atomic_write_text(path, content)

def _format_ts_for_example(seconds: float) -> str:
    ms = int(round(seconds * 1000.0))
    hrs = ms // 3_600_000
    ms %= 3_600_000
    mins = ms // 60_000
    ms %= 60_000
    secs = ms // 1000
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

def _looks_japanese_text(s: str) -> bool:
    for ch in s:
        code = ord(ch)
        if (0x3040 <= code <= 0x309F) or (0x30A0 <= code <= 0x30FF) or (0x4E00 <= code <= 0x9FFF):
            return True
    return False

def build_glossary(turns: List[Turn]) -> List[Dict[str, str]]:
    """
    Build a clean, de-duplicated glossary:
      - Lemmatize (Sudachi -> fugashi -> heuristic fallback)
      - Keep content words only (N/V/Adj/Adv; + proper nouns)
      - Drop particles/auxiliaries/polite/suffix tokens/honorifics
      - Key by lemma (normalized form) so inflection variants collapse
    """
    entries: Dict[str, Dict[str, Union[str, int, float]]] = {}

    def _add_entry(lemma: str, reading_hira: str, example: str, first_ts: float):
        key = _normalize_text_nfc(lemma)
        key = re.sub(r"\s+", "", key)
        if not key:
            return

        # Drop obvious noise
        if key in _JA_DROP_EXACT:
            return
        if key in _JA_POLITE_AUX:
            return
        if (len(key) <= 1) and (not _contains_kanji(key)):
            return

        if key not in entries:
            entries[key] = {
                "term": lemma,
                "hira": reading_hira,
                "roma": _normalize_text_nfc(pykakasi.kakasi().convert(lemma)[0]["hepburn"]) if _HAS_KAKASI else "",
                "count": 1,
                "first_ts": _format_ts_for_example(first_ts),
                "example": example,
            }
        else:
            e = entries[key]
            e["count"] = int(e.get("count", 0)) + 1  # type: ignore
            # fill missing reading if any
            if not e.get("hira") and reading_hira:
                e["hira"] = reading_hira  # type: ignore

    # -------- Path A: Sudachi (best) --------
    if _HAS_SUDACHI:
        for t in turns:
            # keep only JP-looking lines or explicitly flagged as JA
            if not (t.language == "ja" or _looks_japanese_text(t.text)):
                continue
            text = _normalize_text_nfc(t.text or "").strip()
            if not text:
                continue
            morps = _SUDACHI.tokenize(text, _SUDACHI_MODE)

            for m in morps:
                pos = m.part_of_speech()  # tuple, e.g. ('名詞','普通名詞','一般',...)
                top = pos[0] if pos else ""
                sub = pos[1] if len(pos) > 1 else ""

                # POS gate
                if top not in _ALLOWED_POS_TOP:
                    continue
                # drop noun subcats like particles/suffix/pronoun-ish
                if top == "名詞" and sub in _NOUN_SUB_DROP:
                    continue
                # dictionary/normalized form
                lemma = _normalize_text_nfc(m.dictionary_form() or m.normalized_form() or m.surface())
                if lemma in _JA_POLITE_AUX or lemma in _JA_DROP_EXACT:
                    continue

                # reading -> hiragana
                reading = m.reading_form() or ""
                reading_hira = _katakana_to_hiragana(reading)

                # final safety: very short kana-only -> drop (unless proper noun)
                if (len(lemma) <= 1) and (not _contains_kanji(lemma)) and not ("固有名詞" in pos):
                    continue

                _add_entry(lemma, reading_hira, text, t.start)

    # -------- Path B: fugashi (good) --------
    elif _HAS_FUGASHI:
        for t in turns:
            if not (t.language == "ja" or _looks_japanese_text(t.text)):
                continue
            text = _normalize_text_nfc(t.text or "").strip()
            if not text:
                continue
            for m in _FUGASHI(text):
                # fugashi/unidic-lite features vary; be defensive
                pos = m.feature.get("pos1", "") or m.feature.pos1 if hasattr(m.feature, "pos1") else ""
                pos2 = m.feature.get("pos2", "") if isinstance(m.feature, dict) else ""
                top = pos
                if top not in _ALLOWED_POS_TOP:
                    continue
                if top == "名詞" and pos2 in _NOUN_SUB_DROP:
                    continue

                lemma = m.feature.get("lemma", "") or m.feature.get("base_form", "") or m.surface
                lemma = _normalize_text_nfc(lemma)
                if lemma in _JA_POLITE_AUX or lemma in _JA_DROP_EXACT:
                    continue

                reading = m.feature.get("reading", "") or ""
                reading_hira = _katakana_to_hiragana(reading)

                if (len(lemma) <= 1) and (not _contains_kanji(lemma)):
                    continue

                _add_entry(lemma, reading_hira, text, t.start)

    # -------- Path C: Heuristic fallback (pykakasi only) --------
    else:
        for t in turns:
            if not (t.language == "ja" or _looks_japanese_text(t.text)):
                continue
            text = _normalize_text_nfc(t.text or "").strip()
            if not text:
                continue

            # crude "content term" extraction: CJK runs length>=2 or contains kanji
            for m in re.finditer(r"[\u3040-\u30FF\u4E00-\u9FFF]+", text):
                term = _normalize_text_nfc(m.group(0))
                # strip common polite endings (coarse)
                term = re.sub(r"(でしたか|でした|たいです|てください|ましょう|ます|です)$", "", term)
                if term in _JA_POLITE_AUX or term in _JA_DROP_EXACT:
                    continue
                if (len(term) <= 1) and (not _contains_kanji(term)):
                    continue

                # reading/roma via pykakasi if available
                if _HAS_KAKASI:
                    conv = _kks.convert(term)
                    hira = _normalize_text_nfc("".join(x.get("hira","") for x in conv))
                else:
                    hira = ""
                _add_entry(term, hira, text, t.start)

    # Optional Spanish translation for study sheet
    if globals().get("_HAS_ARGOS") and globals().get("_STUDY_TRANSLATE", False):
        for k, e in entries.items():
            term = str(e.get("term", ""))
            try:
                es = translate_ja_to_es(term) or ""
            except Exception:
                es = ""
            e["es"] = es  # type: ignore

    # Sort by frequency desc, then term
    sorted_entries = sorted(entries.values(), key=lambda d: (-int(d["count"]), str(d["term"])))  # type: ignore
    return [
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
    
    # Deduplicate keys that differ only by spacing or Unicode form
    entries = { _normalize_text_nfc(k): v for k, v in entries.items() }

    if globals().get("_HAS_ARGOS") and globals().get("_STUDY_TRANSLATE", False):
        for k, e in entries.items():
            term = e["term"]  # type: ignore
            try:
                es = translate_ja_to_es(term) or ""
            except Exception:
                es = ""
            e["es"] = es  # type: ignore

    sorted_entries = sorted(entries.values(), key=lambda d: (-int(d["count"]), str(d["term"])))  # type: ignore
    return [
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
    Write a human-friendly glossary. One entry per block.
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
        """Ensure JA→EN and EN→ES models are installed (idempotent)."""
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
# Experiment helpers
# -----------------------

def _majority_language(turns: List[Turn]) -> Tuple[str, float]:
    langs = [t.language for t in turns if t.language and t.language != "und"]
    if not langs:
        return "und", 0.0
    c = Counter(langs)
    lang, n = c.most_common(1)[0]
    return lang, n / len(langs)

_STOPWORDS = set("""
a an and the of to in for with on at from as by this that these those is are was were be been being do does did
you your yours we our ours he she it they them i me my mine or but so if then because into out up down about
over under again further once here there when where why how all any both each few more most other some such no nor not
only own same than too very can will just don don't should could would might must
""".split())

def _extract_domain_terms(turns: List[Turn], max_terms: int = 25) -> str:
    text = " ".join(t.text for t in turns)
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text)
    counts = Counter(tok for tok in tokens if tok.lower() not in _STOPWORDS)
    boosted = [(tok, cnt * (1.5 if any(c.isupper() for c in tok) else 1.0)) for tok, cnt in counts.items()]
    boosted.sort(key=lambda x: -x[1])
    terms = [tok for tok, _ in boosted[:max_terms]]
    return ", ".join(terms)

# -----------------------
# JA/ES enforcement helpers
# -----------------------

def _ffmpeg_cut_to_tmp(src_path: str, start: float, end: float) -> str:
    """Cut [start,end] into a temp WAV (mono 16k). Returns path."""
    start = max(0.0, float(start))
    end = max(start, float(end))
    tmp = tempfile.NamedTemporaryFile(prefix="clip_", suffix=".wav", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg","-y","-nostdin",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-i", src_path, "-vn","-ac","1","-ar","16000", tmp.name
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        try: os.unlink(tmp.name)
        except Exception: pass
        raise RuntimeError(f"ffmpeg cut failed: {proc.stderr.decode(errors='ignore')[:400]}")
    return tmp.name

def _force_transcribe_window(shared_model: WhisperModel, audio_path: str,
                             start: float, end: float, lang: str,
                             beam_size: int, condition_on_previous_text: bool,
                             initial_prompt: Optional[str]) -> str:
    """Decode a short window with language forced. Returns concatenated text."""
    path = _ffmpeg_cut_to_tmp(audio_path, start, end)
    try:
        segs, _info = shared_model.transcribe(
            path,
            task="transcribe",
            language=lang,
            beam_size=beam_size,
            vad_filter=False,                # don't re-VAD tiny windows
            word_timestamps=False,
            condition_on_previous_text=condition_on_previous_text,
            initial_prompt=initial_prompt
        )
        txt = " ".join((s.text or "").strip() for s in segs)
        return _normalize_text_nfc(txt)
    finally:
        try: os.unlink(path)
        except Exception: pass

def _speaker_majorities(turns: List[Turn]) -> Dict[str, Tuple[str, float]]:
    """Return {speaker: (majority_lang, frac)}."""
    by_spk = defaultdict(list)
    for t in turns:
        if t.language and t.language != "und":
            by_spk[t.speaker].append(t.language)
    out = {}
    for spk, langs in by_spk.items():
        if not langs:
            out[spk] = ("und", 0.0)
        else:
            c = Counter(langs)
            lang, n = c.most_common(1)[0]
            out[spk] = (lang, n/len(langs))
    return out

_SP_DIAC = re.compile(r"[áéíóúñÁÉÍÓÚÑ¿¡]")

def _guess_target_lang(text: str) -> Optional[str]:
    """Heuristic guess for JA vs ES."""
    if looks_japanese(text):
        return "ja"
    if _SP_DIAC.search(text):
        return "es"
    lbl, score = langid.classify(text or "")
    if lbl in ("ja","es") and score >= 0.7:
        return lbl
    return None

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
                    help="Optional explicit revision for the diarization pipeline.")
    ap.add_argument("--hf-cache-dir", default=None,
                    help="Optional Hugging Face cache directory to reduce repeated downloads (sets HF_HOME).")
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

    # Experiments
    ap.add_argument("--n-experiments", type=int, default=1,
                    help="Run N transcription experiments with heuristic tweaks. Default: 1 (baseline only).")
    ap.add_argument("--experiment-scheme", choices=["auto", "accuracy", "speed"], default="auto",
                    help="Heuristic preset priority: accuracy/speed/auto.")

    # JA/ES enforcement / rescue
    ap.add_argument("--allowed-languages", default="ja,es",
                    help="Comma-separated ISO 639-1 codes allowed for output. Default: ja,es.")
    ap.add_argument("--redecode-offlang", action="store_true",
                    help="If a turn language is not in --allowed-languages, re-decode that window with JA/ES and keep the better one.")
    ap.add_argument("--redecode-pad", type=float, default=0.25,
                    help="Seconds of audio added on each side when re-decoding a turn window. Default: 0.25.")
    ap.add_argument("--redecode-max-seg-sec", type=float, default=30.0,
                    help="Max seconds per turn eligible for re-decode. Default: 30.")
    ap.add_argument("--redecode-max-total-sec", type=float, default=180.0,
                    help="Budget cap (s) of audio to re-decode across all turns. Default: 180.")
    ap.add_argument("--enforce-lang-by-speaker", action="store_true",
                    help="Lock each speaker to their majority language (if JA/ES and confident) and re-decode their turns.")
    ap.add_argument("--enforce-lang-threshold", type=float, default=0.7,
                    help="Minimum fraction for a speaker's majority language to qualify for locking. Default: 0.7.")

    return ap.parse_args()

# -----------------------
# Main
# -----------------------

def main():
    args = parse_args()
    _t0_total_script = time.perf_counter()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    if not args.verbose:
        logging.disable(logging.INFO)
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

    # ---- Early FFmpeg/FFprobe guards (fail fast)
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

    # Optional denoise/preprocess — generate once to reuse across runs
    temp_denoised = None
    denoised_available = False
    try:
        if args.ffmpeg_denoise_model:
            if args.verbose:
                logging.info("[Preprocess] Denoising with FFmpeg arnndn")
            pbar_dn = tqdm(total=1, desc="Preparing denoised input", disable=not (args.verbose or args.progress))
            temp_denoised = ffmpeg_denoise(str(in_path), args.ffmpeg_denoise_model, target_sr=16000)
            denoised_available = True
            pbar_dn.update(1); pbar_dn.close()
    except Exception as e:
        logging.warning(f"[Preprocess] Denoise failed: {e}. Continuing with original input only.")
        temp_denoised = None
        denoised_available = False

    raw_input = str(in_path)
    default_working_input = temp_denoised if denoised_available else raw_input

    # Device & shared Whisper model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shared_model = WhisperModel(args.model_size, device=device, compute_type=args.compute_type)

    # Media duration cache per audio variant (for pbar)
    duration_cache: Dict[str, Optional[float]] = {}
    def _get_duration(p: str) -> Optional[float]:
        if p not in duration_cache:
            duration_cache[p] = probe_duration_ffprobe(p)
        return duration_cache[p]

    # Diarization cache per audio variant
    diar_cache: Dict[str, Annotation] = {}
    def _get_diarization(audio_variant_path: str) -> Annotation:
        if args.skip_diarization or (not args.hf_token):
            ann = Annotation()
            return ann  # Fallback handled later
        if audio_variant_path in diar_cache:
            return diar_cache[audio_variant_path]
        # Build diarization input based on --preload-audio
        if args.preload_audio:
            audio_input: Union[str, dict] = load_audio_ta(audio_variant_path)
            if args.verbose:
                logging.info("[Main] --preload-audio requested; bypassing FFmpeg CLI for diarization")
        else:
            audio_input = audio_variant_path
        pipeline_name = args.diarization_pipeline
        if args.diarization_revision:
            if "@" in pipeline_name:
                pipeline_name = pipeline_name.split("@", 1)[0]
            pipeline_name = f"{pipeline_name}@{args.diarization_revision}"
        pbar_dz = tqdm(total=1, desc="Diarizing", disable=not (args.verbose or args.progress))
        try:
            diar = diarize_speakers(
                audio_input=audio_input,
                hf_token=args.hf_token,
                pipeline_name=pipeline_name,
                num_speakers=args.num_speakers,
                verbose=args.verbose
            )
        except Exception as e:
            logging.warning(f"[Main] Diarization failed ({e.__class__.__name__}: {e}); falling back to single-speaker.")
            diar = Annotation()
        finally:
            try:
                pbar_dz.update(1); pbar_dz.close()
            except Exception:
                pass
        if hasattr(diar, "annotation"):
            diar = diar.annotation
        diar_cache[audio_variant_path] = diar
        return diar

    # Outputs collector
    base = in_path.with_suffix("")
    fmts = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    all_outputs: Dict[int, Dict[str, Path]] = {}
    summary_rows: List[Dict[str, Union[str, int, float, None, bool]]] = []

    def _language_consistency(turns: List[Turn]) -> Tuple[str, float]:
        lang, frac = _majority_language(turns)
        return lang, frac

    def _compute_metrics(asr_segments: List[Dict], turns: List[Turn]) -> Dict[str, Union[int, float, None, Dict]]:
        total_words = sum(len(s["text"].split()) for s in asr_segments)
        spoken_seconds = sum(max(0.0, (s["end"] - s["start"])) for s in asr_segments)
        audio_seconds = max((s["end"] for s in asr_segments), default=0.0)
        wpm_full = (total_words / (audio_seconds / 60.0)) if audio_seconds > 0 else None
        wpm_spoken = (total_words / (spoken_seconds / 60.0)) if spoken_seconds > 0 else None

        # per-speaker stats
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
        maj_lang, lang_frac = _language_consistency(turns)
        return {
            "audio_seconds_estimate": round(audio_seconds, 3),
            "spoken_seconds_sum": round(spoken_seconds, 3),
            "total_words": int(total_words),
            "wpm_over_full_audio": (round(wpm_full, 2) if wpm_full is not None else None),
            "wpm_over_spoken_time": (round(wpm_spoken, 2) if wpm_spoken is not None else None),
            "per_speaker": speaker_metrics,
            "majority_language": maj_lang,
            "language_consistency": round(float(lang_frac), 4) if lang_frac is not None else None,
        }

    def _write_outputs(exp_id: int, turns: List[Turn], meta: Dict):
        settings_header = _format_settings_header(meta)
        run_suffix = f"{args.suffix}_exp{exp_id:02d}"
        outputs: Dict[str, Path] = {}
        # Emit machine-readable meta **only** when JSON is requested
        if "json" in fmts:
            meta_path = Path(f"{base}{run_suffix}.meta.json")
            write_meta_json(meta, meta_path)
            outputs["meta"] = meta_path
            p = Path(f"{base}{run_suffix}.json")
            write_json(turns, p)
            outputs["json"] = p
        if "srt" in fmts:
            p = Path(f"{base}{run_suffix}.srt"); write_srt(turns, p, settings_header); outputs["srt"] = p
        if "vtt" in fmts:
            p = Path(f"{base}{run_suffix}.vtt"); write_vtt(turns, p, settings_header); outputs["vtt"] = p
        if "txt" in fmts:
            p = Path(f"{base}{run_suffix}.txt"); write_txt(turns, p, settings_header); outputs["txt"] = p
        if "csv" in fmts:
            p = Path(f"{base}{run_suffix}.csv"); write_csv(turns, p, settings_header); outputs["csv"] = p
        if "study" in fmts:
            p = Path(f"{base}{run_suffix}.study.txt")
            write_study(turns, p, settings_header,
                        include_hiragana=not args.study_no_hiragana,
                        include_romaji=not args.study_no_romaji)
            outputs["study"] = p
        if args.glossary:
            g = build_glossary(turns)
            p = Path(f"{base}{run_suffix}.glossary.txt")
            write_glossary(g, p, settings_header)
            outputs["glossary"] = p
        all_outputs[exp_id] = outputs

    # -----------------------
    # Single experiment runner (includes JA/ES enforcement)
    # -----------------------
    def run_experiment(exp_id: int, params: Dict) -> Tuple[List[Turn], Dict]:
        """
        Run a single experiment; returns (turns, meta).
        """
        # Choose audio variant for this run
        using_denoised = bool(params.get("use_denoised", denoised_available))
        working_input = (temp_denoised if (using_denoised and denoised_available) else raw_input)
        media_duration = _get_duration(working_input)

        # --- Transcribe
        _t0_asr = time.perf_counter()
        asr_segments, info_fields = transcribe_audio(
            working_input,
            model_size=args.model_size,
            compute_type=args.compute_type,
            beam_size=int(params.get("beam_size", args.beam_size)),
            vad_filter=bool(params.get("vad_filter", not args.no_vad)),
            word_timestamps=not args.no_word_timestamps,
            condition_on_previous_text=bool(params.get("condition_on_previous_text", not args.no_condition_on_previous_text)),
            language=params.get("language") or (args.language or None),
            patience=params.get("patience", args.patience),
            compression_ratio_threshold=params.get("compression_ratio_threshold", args.compression_ratio_threshold),
            no_speech_threshold=params.get("no_speech_threshold", args.no_speech_threshold),
            initial_prompt=params.get("initial_prompt", args.initial_prompt),
            verbose=args.verbose,
            progress=(args.verbose or args.progress),
            total_duration=media_duration,
            _shared_model=shared_model,
        )
        asr_seconds = time.perf_counter() - _t0_asr

        # --- Diarization (reuse per variant)
        _t0_diar = time.perf_counter()
        diar = _get_diarization(working_input)
        diar_seconds = time.perf_counter() - _t0_diar

        # Fallback to single speaker if needed
        if diar is None or (hasattr(diar, "labels") and len(list(diar.labels())) == 0):
            diar = Annotation()
            diar[Segment(0.0, max((s['end'] for s in asr_segments), default=0.0))] = "Speaker 1"
        elif not hasattr(diar, "itersegments"):
            if hasattr(diar, "annotation"):
                diar = diar.annotation

        # --- Build & merge turns
        turns = build_turns(asr_segments, diar, show_progress=(args.verbose or args.progress))
        turns = merge_adjacent(turns, max_gap=float(args.merge_gap))

        # --- JA/ES Enforcement / off-language rescue (optional)
        rescued_turns_count = 0
        redecode_seconds_spent = 0.0

        if args.redecode_offlang or args.enforce_lang_by_speaker:
            allowed = {c.strip().lower() for c in (args.allowed_languages or "").split(",") if c.strip()}
            if not allowed:
                allowed = {"ja", "es"}

            # Speaker majorities (if enabled)
            spk_major = _speaker_majorities(turns) if args.enforce_lang_by_speaker else {}

            def _decide_target_for_turn(t: Turn) -> Optional[str]:
                # If already allowed, no action
                if t.language in allowed:
                    return None
                # Speaker majority, if confident and in allowed set
                if args.enforce_lang_by_speaker:
                    maj, frac = spk_major.get(t.speaker, ("und", 0.0))
                    if maj in allowed and frac >= float(args.enforce_lang_threshold):
                        return maj
                # Quick guess from text
                g = _guess_target_lang(t.text)
                if g in allowed:
                    return g
                # Last resort: try both and pick best
                return "both"

            def _langid_score(txt: str, expect: str) -> float:
                if not txt.strip():
                    return -1.0
                lbl, sc = langid.classify(txt)
                return sc if lbl == expect else -sc  # penalize mismatches

            budget_total = float(args.redecode_max_total_sec)
            max_seg = float(args.redecode_max_seg_sec)
            pad = max(0.0, float(args.redecode_pad))

            new_turns: List[Turn] = []
            for t in turns:
                target = _decide_target_for_turn(t) if args.redecode_offlang or args.enforce_lang_by_speaker else None
                if not target:
                    new_turns.append(t)
                    continue

                seg_len = max(0.0, t.end - t.start)
                if seg_len <= 0.0 or seg_len > max_seg:
                    new_turns.append(t)
                    continue
                if (redecode_seconds_spent + seg_len) > budget_total:
                    new_turns.append(t)
                    continue

                s0 = max(0.0, t.start - pad)
                e0 = t.end + pad

                if target in ("ja", "es"):
                    txt_new = _force_transcribe_window(
                        shared_model, working_input, s0, e0, target,
                        beam_size=int(params.get("beam_size", args.beam_size)),
                        condition_on_previous_text=bool(params.get("condition_on_previous_text", not args.no_condition_on_previous_text)),
                        initial_prompt=params.get("initial_prompt", args.initial_prompt)
                    )
                    if txt_new.strip():
                        t.text = txt_new
                        t.language, t.lang_score = detect_language_for_text(txt_new)
                        rescued_turns_count += 1
                        redecode_seconds_spent += seg_len
                    new_turns.append(t)
                    continue

                # target == "both": try JA and ES, pick by langid score (+small JP-char bonus)
                best_txt = None; best_lang = None; best_score = -1e9
                for lang_try in ("ja", "es"):
                    txt_try = _force_transcribe_window(
                        shared_model, working_input, s0, e0, lang_try,
                        beam_size=int(params.get("beam_size", args.beam_size)),
                        condition_on_previous_text=bool(params.get("condition_on_previous_text", not args.no_condition_on_previous_text)),
                        initial_prompt=params.get("initial_prompt", args.initial_prompt)
                    )
                    score = _langid_score(txt_try, lang_try)
                    if lang_try == "ja" and looks_japanese(txt_try):
                        score += 0.25  # tie-break toward JP if CJK present
                    if score > best_score:
                        best_score = score; best_txt = txt_try; best_lang = lang_try

                if best_txt and best_txt.strip():
                    t.text = best_txt
                    t.language, t.lang_score = detect_language_for_text(best_txt)
                    rescued_turns_count += 1
                    redecode_seconds_spent += seg_len
                new_turns.append(t)

            turns = new_turns

        # --- Metrics (after any rescue)
        metrics = _compute_metrics(asr_segments, turns)

        # --- Meta per run
        total_seconds = asr_seconds + diar_seconds
        meta = {
            "experiment_id": exp_id,
            "experiment_params": {
                "beam_size": int(params.get("beam_size", args.beam_size)),
                "vad_filter": bool(params.get("vad_filter", not args.no_vad)),
                "condition_on_previous_text": bool(params.get("condition_on_previous_text", not args.no_condition_on_previous_text)),
                "language": params.get("language") or (args.language or None),
                "patience": params.get("patience", args.patience),
                "compression_ratio_threshold": params.get("compression_ratio_threshold", args.compression_ratio_threshold),
                "no_speech_threshold": params.get("no_speech_threshold", args.no_speech_threshold),
                "initial_prompt": bool(params.get("initial_prompt", args.initial_prompt)),
                "use_denoised": using_denoised and denoised_available,
                # Language rescue knobs
                "allowed_languages": (args.allowed_languages or "ja,es"),
                "redecode_offlang": bool(args.redecode_offlang),
                "redecode_pad": float(args.redecode_pad),
                "redecode_max_seg_sec": float(args.redecode_max_seg_sec),
                "redecode_max_total_sec": float(args.redecode_max_total_sec),
                "enforce_lang_by_speaker": bool(args.enforce_lang_by_speaker),
                "enforce_lang_threshold": float(args.enforce_lang_threshold),
            },
            "input": str(in_path),
            "used_input": working_input,
            "device": device,
            "whisper_model": args.model_size,
            "compute_type": args.compute_type,
            "word_timestamps": not args.no_word_timestamps,
            "merge_gap": float(args.merge_gap),
            "diarization_pipeline": args.diarization_pipeline,
            "diarization_revision": args.diarization_revision,
            "hf_cache_dir": args.hf_cache_dir,
            "num_speakers": args.num_speakers,
            "skip_diarization": bool(args.skip_diarization or (not args.hf_token)),
            "pyannote_device": device,
            "study_translate": not args.no_study_translate,
            "info_fields": info_fields,
            "metrics": {
                "transcribe_seconds": round(asr_seconds, 3),
                "diarization_seconds": round(diar_seconds, 3),
                "total_seconds": round(total_seconds, 3),
                **metrics,
                # Rescue stats
                "rescued_turns": int(rescued_turns_count),
                "redecode_seconds_budget": float(args.redecode_max_total_sec),
                "redecode_seconds_spent": round(redecode_seconds_spent, 3),
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

        # Make study translate flag visible to writer
        globals()["_STUDY_TRANSLATE"] = not args.no_study_translate

        # --- Write per-run outputs
        _write_outputs(exp_id, turns, meta)

        # --- Add to experiments summary
        summary_rows.append({
            "experiment_id": exp_id,
            "beam_size": int(params.get("beam_size", args.beam_size)),
            "vad_filter": bool(params.get("vad_filter", not args.no_vad)),
            "condition_on_previous_text": bool(params.get("condition_on_previous_text", not args.no_condition_on_previous_text)),
            "language": params.get("language") or (args.language or None),
            "patience": params.get("patience", args.patience),
            "compression_ratio_threshold": params.get("compression_ratio_threshold", args.compression_ratio_threshold),
            "no_speech_threshold": params.get("no_speech_threshold", args.no_speech_threshold),
            "initial_prompt_len": len(params.get("initial_prompt", args.initial_prompt) or "") or 0,
            "use_denoised": using_denoised and denoised_available,
            "transcribe_seconds": meta["metrics"]["transcribe_seconds"],
            "diarization_seconds": meta["metrics"]["diarization_seconds"],
            "total_seconds": meta["metrics"]["total_seconds"],
            "audio_seconds_estimate": meta["metrics"]["audio_seconds_estimate"],
            "spoken_seconds_sum": meta["metrics"]["spoken_seconds_sum"],
            "total_words": meta["metrics"]["total_words"],
            "wpm_over_full_audio": meta["metrics"]["wpm_over_full_audio"],
            "wpm_over_spoken_time": meta["metrics"]["wpm_over_spoken_time"],
            "majority_language": meta["metrics"]["majority_language"],
            "language_consistency": meta["metrics"]["language_consistency"],
            # Rescue columns
            "rescued_turns": int(rescued_turns_count),
            "redecode_seconds_spent": round(redecode_seconds_spent, 3),
        })

        return turns, meta

    # -----------------------
    # Build experiment plan
    # -----------------------
    n = max(1, int(args.n_experiments))
    baseline_params = {
        "beam_size": args.beam_size,
        "vad_filter": (not args.no_vad),
        "condition_on_previous_text": (not args.no_condition_on_previous_text),
        "language": (args.language or None),
        "patience": args.patience,
        "compression_ratio_threshold": args.compression_ratio_threshold,
        "no_speech_threshold": args.no_speech_threshold,
        "initial_prompt": args.initial_prompt,
        "use_denoised": bool(denoised_available),  # baseline uses denoised if available
    }

    experiment_params_list: List[Dict] = [baseline_params]  # exp 1 is baseline

    # Run baseline first (so later heuristics can use signals from it)
    baseline_turns, baseline_meta = run_experiment(1, baseline_params)

    # Heuristics pool (ordered)
    base_used_denoised = baseline_params["use_denoised"]
    majority_lang, lang_frac = _majority_language(baseline_turns)
    auto_prompt = _extract_domain_terms(baseline_turns, max_terms=25)

    H_list = []

    # H1: wider beam + patience
    H_list.append({**baseline_params, "beam_size": 8, "patience": (baseline_params["patience"] or 1.2)})

    # H2: hallucination guard(s)
    H_list.append({**baseline_params,
                   "compression_ratio_threshold": (baseline_params["compression_ratio_threshold"] or 2.3),
                   "no_speech_threshold": max(0.6, baseline_params["no_speech_threshold"] or 0.6)})

    # H3: VAD off (recover clipped words)
    H_list.append({**baseline_params, "vad_filter": False})

    # H4: language lock to majority (if not already locked and confident enough)
    if (not baseline_params["language"]) and (majority_lang != "und") and (lang_frac >= 0.6):
        H_list.append({**baseline_params, "language": majority_lang})

    # H5: disable condition_on_previous_text (topic drift cases)
    H_list.append({**baseline_params, "condition_on_previous_text": False})

    # H6: switch raw/denoised if possible
    if denoised_available:
        H_list.append({**baseline_params, "use_denoised": (not base_used_denoised)})

    # H7: relaxed no_speech threshold
    H_list.append({**baseline_params,
                   "no_speech_threshold": min(0.45, baseline_params["no_speech_threshold"] or 0.45)})

    # H8: smaller beam (sometimes helps choppy speakers)
    H_list.append({**baseline_params, "beam_size": 3, "patience": None})

    # H9: domain priming with auto prompt
    if auto_prompt:
        short_prompt = ", ".join(auto_prompt.split(", ")[:20])
        H_list.append({**baseline_params, "initial_prompt": short_prompt})

    # Scheme re-ordering
    if args.experiment_scheme == "accuracy":
        order = [0, 1, 3, 5, 2, 6, 7, 8]  # prefer wider beam, guards, language lock, denoise swap
    elif args.experiment_scheme == "speed":
        order = [2, 7, 8, 4, 6, 1, 0, 5]  # cheap tweaks first
    else:  # auto
        order = list(range(len(H_list)))
    H_list = [H_list[i] for i in order if i < len(H_list)]

    # Fill remaining slots with mild variations if user requested more
    extra_needed = max(0, n - 1 - len(H_list))
    if extra_needed > 0:
        alt_beams = [7, 6, 4, 2]
        for b in alt_beams:
            if extra_needed <= 0:
                break
            H_list.append({**baseline_params, "beam_size": b})
            extra_needed -= 1

    # Take only up to (n-1) heuristics
    H_list = H_list[:max(0, n - 1)]

    # Run the remaining experiments
    for i, params in enumerate(H_list, start=2):
        run_experiment(i, params)

    # ---- Experiments summary CSV
    summary_csv_path = Path(f"{base}{args.suffix}_experiments.csv")
    df = pd.DataFrame(summary_rows)
    _atomic_write_text(summary_csv_path, df.to_csv(index=False))

    # Print human summary
    print("\nDone. Outputs by experiment:")
    for exp_id, outs in sorted(all_outputs.items()):
        print(f"  EXP {exp_id:02d}:")
        for k, p in outs.items():
            print(f"    - {k.upper():8s} {p}")
    print(f"\nExperiments summary CSV: {summary_csv_path}")

    # Global timing
    print("\nSummary (script-level):")
    print(f"  Experiments run : {len(all_outputs)}")
    print(f"  Total runtime   : {time.perf_counter() - _t0_total_script:.2f}s")

    # Cleanup temp
    if temp_denoised:
        try: os.unlink(temp_denoised)
        except Exception: pass

if __name__ == "__main__":
    main()

# (end)
