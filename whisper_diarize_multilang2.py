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
"""

"""
--------------- ENVIRONMENT SETUP --------------- 
# --- fresh start
conda deactivate
conda env remove --name py312 -y
conda clean --all -y

conda create -n py312 python=3.12 -y
conda activate py312

# --- multimedia dependencies
conda install "ffmpeg<8" -y

# --- ✅ install PyTorch stack with matching CUDA runtime (no full toolkit)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# --- diarization and audio utils
pip install pyannote.audio pykakasi faster-whisper argostranslate langid


# conda install -c conda-forge cudnn
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
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Union

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning, module="ctranslate2",)
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
warnings.filterwarnings("ignore", module="pyannote.audio.utils.reproducibility", category=UserWarning)
warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom is <= 0")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pykakasi")


import langid
import pandas as pd
from tqdm import tqdm

# --- pykakasi: new API (no deprecation warnings)
try:
    from pykakasi import kakasi
    _HAS_KAKASI = True
    _k_hira = kakasi()
    _k_hira.setMode("J", "H")  # Kanji -> Hiragana
    _k_hira.setMode("K", "H")  # Katakana -> Hiragana
    _k_hira.setMode("H", "H")  # Hiragana -> Hiragana
    _conv_hira = _k_hira.getConverter()

    _k_romaji = kakasi()
    _k_romaji.setMode("J", "a")  # Kanji -> Romaji
    _k_romaji.setMode("K", "a")  # Katakana -> Romaji
    _k_romaji.setMode("H", "a")  # Hiragana -> Romaji
    _conv_romaji = _k_romaji.getConverter()

    def jp_with_readings(text: str):
        tokens = _k_romaji.convert(text)
        hira = " ".join(tok["hira"] for tok in _k_hira.convert(text))
        roma = " ".join(tok["hepburn"] for tok in _k_romaji.convert(text))
        return hira.strip(), roma.strip()
except Exception:
    _HAS_KAKASI = False
    def jp_with_readings(_): return "", ""

# --- Whisper, Pyannote
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.core import Annotation

# We’ll try to use torchcodec; if not present or fails at runtime, we fallback to preloaded audio
def _has_torchcodec() -> bool:
    try:
        import torchcodec  # noqa: F401
        return True
    except Exception:
        return False

# --- torch/torchaudio for preload fallback + device detection
import torch
import torchaudio


# -----------------------
# Utilities
# -----------------------

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
    # ffmpeg -i input -f f32le -ac 1 -ar 16000 pipe:1
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0", "-i", path,
        "-vn", "-ac", "1", "-ar", str(target_sr),
        "-f", "f32le", "pipe:1"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed (code {proc.returncode}): {proc.stderr.decode(errors='ignore')[:500]}")
    # Convert raw bytes → float32 mono tensor
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise RuntimeError("ffmpeg produced no audio samples.")
    wav = torch.from_numpy(audio).unsqueeze(0)  # (1, T) mono
    return {"waveform": wav, "sample_rate": target_sr}


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
    verbose: bool = False,
) -> List[Dict]:
    # Auto device: CUDA if available, else CPU — independent of compute_type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        logging.info(f"[Whisper] device={device} model={model_size} compute_type={compute_type} beam={beam_size} vad={vad_filter}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments_gen, info = model.transcribe(
        audio_path,
        task="transcribe",      # keep original languages; use "translate" to force English
        language=None,          # auto-detect; Whisper handles code-switching
        beam_size=beam_size,
        vad_filter=vad_filter,
        word_timestamps=word_timestamps,
        condition_on_previous_text=condition_on_previous_text,
    )

    segments = []
    for seg in segments_gen:
        segments.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})

    if verbose:
        logging.info(f"[Whisper] primary language: {getattr(info, 'language', None)} "
                     f"(p={getattr(info, 'language_probability', None)}) | segments={len(segments)}")
    return segments


def diarize_speakers(
    audio_input: Union[str, dict],
    hf_token: str,
    pipeline_name: str = "pyannote/speaker-diarization-3.1",
    verbose: bool = False,
) -> Annotation:
    """
    Diarize speakers using Pyannote.
    Compatible with both the new DiarizeOutput (v3.1+) and older Annotation formats.
    Automatically tries TorchCodec first, then torchaudio fallback.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not hf_token:
        raise RuntimeError("Hugging Face token not found. Set HUGGINGFACE_TOKEN env var or pass --hf-token.")

    if verbose:
        logging.info(f"[Pyannote] loading pipeline: {pipeline_name}")

    # Parse model_id and revision from the pipeline_name string
    model_id = pipeline_name
    revision = None
    if "@" in model_id:
        model_id, revision = model_id.split("@", 1)
        if verbose:
            logging.info(f"[Pyannote] using model_id='{model_id}' and revision='{revision}'")

    # honor the user-provided pipeline (default: 3.1)
    pipeline = Pipeline.from_pretrained(
        model_id,
        revision=revision,  # may be None
        token=hf_token
    )
    
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

    # 1) Try FFmpeg CLI path first (decode-on-the-fly to tensor)
    if try_ffmpeg_first:
        try:
            if verbose:
                logging.info("[Pyannote] trying FFmpeg CLI decode path")
            audio_dict = load_audio_ffmpeg(audio_input)  # mono float32 16k
            result = pipeline(audio_dict)
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
    result = pipeline(audio_dict)
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
            # `turn` is a pyannote.core.Segment
            ann[turn] = str(speaker)

    else:
        raise TypeError(
            f"Unexpected diarization type: {type(diarization)}. "
            "Expected Annotation, or an object with .annotation or .speaker_diarization."
        )

    # Now proceed uniformly with an Annotation
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


def build_turns(asr_segments: List[Dict], diarization: Annotation, verbose: bool = False) -> List[Turn]:
    speaker_assigned = assign_speakers_to_asr(asr_segments, diarization)
    turns: List[Turn] = []
    it = tqdm(speaker_assigned, desc="Tagging language / merging", disable=not verbose)
    for seg, spk in it:
        code, score = detect_language_for_text(seg["text"])
        turns.append(Turn(
            start=float(seg["start"]),
            end=float(seg["end"]),
            speaker=spk,
            text=seg["text"],
            language=code,
            lang_score=score,
        ))
    turns.sort(key=lambda t: t.start)
    mapping = relabel_speakers([t.speaker for t in turns])
    for t in turns:
        t.speaker = mapping.get(t.speaker, t.speaker)
    return turns


# -----------------------
# Writers
# -----------------------

# -----------------------
# Offline translation (Argos Translate)
# -----------------------
try:
    import argostranslate.package, argostranslate.translate
    _HAS_ARGOS = True

    def _ensure_argos_models():
        """
        Ensure the JA→EN and EN→ES models are installed.
        Safe to call multiple times (no double installs).
        """
        try:
            # Refresh package index and check installed packages
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


def write_json(turns: List[Turn], path: Path):
    data = [asdict(t) for t in turns]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def write_srt(turns: List[Turn], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for i, t in enumerate(turns, start=1):
            f.write(f"{i}\n")
            f.write(f"{hms(t.start)} --> {hms(t.end)}\n")
            lang_tag = t.language if t.language else "und"
            f.write(f"{t.speaker} [{lang_tag}]: {t.text}\n\n")

def write_vtt(turns: List[Turn], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for t in turns:
            f.write(f"{vtt_ts(t.start)} --> {vtt_ts(t.end)}\n")
            lang_tag = t.language if t.language else "und"
            f.write(f"{t.speaker} [{lang_tag}]: {t.text}\n\n")

def write_txt(turns: List[Turn], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for t in turns:
            f.write(f"[{hms(t.start)[:-4]}] {t.speaker} ({t.language}): {t.text}\n")

def write_csv(turns: List[Turn], path: Path):
    rows = [{
        "start": t.start,
        "end": t.end,
        "speaker": t.speaker,
        "language": t.language,
        "lang_score": t.lang_score,
        "text": t.text
    } for t in turns]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")

def write_study(turns: List[Turn], path: Path, include_hiragana: bool = True, include_romaji: bool = True):
    """
    Study-friendly TXT:
    - Keeps time, speaker, language
    - For Japanese lines: adds kana + romaji hints if pykakasi available
    """
    with open(path, "w", encoding="utf-8") as f:
        header = "# Study Transcript (speaker order, with Japanese readings)\n\n"
        if not _HAS_KAKASI and any((t.language == "ja" or looks_japanese(t.text)) for t in turns):
            header += "(Note: pykakasi not installed — hiragana/romaji hints disabled)\n\n"
        f.write(header)
        for t in turns:
            f.write(f"[{hms(t.start)[:-4]}] {t.speaker} ({t.language})\n")
            f.write(t.text + "\n")
            if (t.language == "ja" or looks_japanese(t.text)) and _HAS_KAKASI:
                hira, roma = jp_with_readings(t.text)
                if include_hiragana and hira.strip():
                    f.write(f"〔hiragana〕 {hira}\n")
                if include_romaji and roma.strip():
                    f.write(f"〔romaji〕 {roma}\n")
            # Optional offline Spanish translation (Argos) for Japanese lines
            if (t.language == "ja" or looks_japanese(t.text)) and globals().get("_HAS_ARGOS") and globals().get("_STUDY_TRANSLATE", False):
                es = translate_ja_to_es(t.text)
                if es:
                    f.write(f"〔español〕 {es}\n")
            f.write("\n")


# -----------------------
# CLI
# -----------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Diarize speakers + transcribe + per-segment language ID using Whisper & Pyannote. Adds study output."
    )
    ap.add_argument("input", help="Path to audio/video file (ffmpeg-compatible).")
    ap.add_argument(
        "--formats",
        default="json,srt,txt",
        help="Comma-separated outputs: json,srt,vtt,txt,csv,study (default: json,srt,txt)"
    )
    ap.add_argument(
        "--suffix",
        default="_whisper_diarized",
        help="Suffix added to output filenames (before extension)."
    )
    ap.add_argument("--model-size", default="large-v3", help="Whisper size: tiny/base/small/medium/large-v3 (default: large-v3)")
    ap.add_argument("--compute-type", default="float16", help="Whisper compute_type: float16,int8_float16,int8,float32")
    ap.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding.")
    ap.add_argument("--no-vad", action="store_true", help="Disable VAD filter.")
    ap.add_argument("--hf-token", default=os.environ.get("HUGGINGFACE_TOKEN", ""), help="Hugging Face token.")
    ap.add_argument("--diarization-pipeline", default="pyannote/speaker-diarization-3.1",
                    help="Pyannote pipeline repo id (default: pyannote/speaker-diarization-3.1).")
    ap.add_argument("--preload-audio", action="store_true",
                    help="Force preloaded-audio (torchaudio) path for diarization (bypass TorchCodec).")
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose logging + progress bars.")
    # decoding controls (default ON; add flags to disable)
    ap.add_argument("--no-word-timestamps", action="store_true", help="Disable word-level timestamps (defaults to ON).")
    ap.add_argument("--no-condition-on-previous-text", action="store_true",
                    help="Disable conditioning on previous text (defaults to ON).")
    # study options
    ap.add_argument("--study-no-hiragana", action="store_true", help="Omit hiragana line in study output.")
    ap.add_argument("--study-no-romaji", action="store_true", help="Omit romaji line in study output.")
    ap.add_argument("--study-translate", action="store_true",
                    help="Add Spanish translation for Japanese segments (offline via Argos Translate).")
    return ap.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    in_path = Path(args.input)
    if not in_path.exists():
        logging.error(f"Input not found: {in_path}")
        sys.exit(1)

    # Transcribe (Whisper)
    asr_segments = transcribe_audio(
        str(in_path),
        model_size=args.model_size,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        vad_filter=not args.no_vad,
        word_timestamps=not args.no_word_timestamps,
        condition_on_previous_text=not args.no_condition_on_previous_text,
        verbose=args.verbose
    )

    # Decide diarization input
    if args.preload_audio:
        audio_input: Union[str, dict] = load_audio_ta(str(in_path))
        if args.verbose:
            logging.info("[Main] --preload-audio requested; bypassing TorchCodec")
    else:
        audio_input = str(in_path)

    # Diarize (Pyannote) with token and pipeline name; token passed as named argument as requested
    diar = diarize_speakers(
        audio_input=audio_input,
        hf_token=args.hf_token,
        pipeline_name=args.diarization_pipeline,
        verbose=args.verbose
    )

    # Unwrap DiarizeOutput if needed
    if hasattr(diar, "annotation"):
        diar = diar.annotation

    # Merge + per-segment language ID
    turns = build_turns(asr_segments, diar, verbose=args.verbose)

    # Outputs — same directory as input
    base = in_path.with_suffix("")  # drop extension
    fmts = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    out_paths = {}

    # Make study translation flag visible to writer without changing its signature
    globals()["_STUDY_TRANSLATE"] = bool(args.study_translate)

    if "json" in fmts:
        p = Path(f"{base}{args.suffix}.json"); write_json(turns, p); out_paths["json"] = p
    if "srt" in fmts:
        p = Path(f"{base}{args.suffix}.srt"); write_srt(turns, p); out_paths["srt"] = p
    if "vtt" in fmts:
        p = Path(f"{base}{args.suffix}.vtt"); write_vtt(turns, p); out_paths["vtt"] = p
    if "txt" in fmts:
        p = Path(f"{base}{args.suffix}.txt"); write_txt(turns, p); out_paths["txt"] = p
    if "csv" in fmts:
        p = Path(f"{base}{args.suffix}.csv"); write_csv(turns, p); out_paths["csv"] = p
    if "study" in fmts:
        p = Path(f"{base}{args.suffix}.study.txt")
        write_study(turns, p, include_hiragana=not args.study_no_hiragana, include_romaji=not args.study_no_romaji)
        out_paths["study"] = p

    if args.verbose:
        for k, p in out_paths.items():
            logging.info(f"Wrote {k.upper()}: {p}")

    print("Done. Outputs:")
    for k, p in out_paths.items():
        print(f"  - {k.upper()}: {p}")


if __name__ == "__main__":
    main()


# C:\Users\epena\AppData\Roaming\Python\Python311\site-packages\~orch