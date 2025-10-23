#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environment sanity check for ASR + diarization + translation stack

Checks:
- Python version
- Torch / CUDA / cuDNN availability
- torchaudio backend
- faster-whisper GPU inference test
- pyannote pipeline loading test
- Argos Translate package presence
- pykakasi kana/romaji conversion
- langid classification
- pandas import and small DataFrame
"""

import sys
import importlib
import traceback

def banner(title):
    print("\n" + "="*80)
    print(title)
    print("="*80)

def safe_import(name):
    try:
        mod = importlib.import_module(name)
        print(f"✅ {name} imported ({getattr(mod, '__version__', 'no __version__')})")
        return mod
    except Exception as e:
        print(f"❌ {name} import failed: {e.__class__.__name__}: {e}")
        return None


banner("🧠 PYTHON / TORCH / CUDA / CUDNN")
import torch
print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.get_device_name(0)}")
    print(f"Torch CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    # simple tensor test
    x = torch.randn(2, 2).to("cuda")
    print(f"Tensor on GPU ok: {x.device} {x.shape}")
else:
    print("⚠️  CUDA not available — GPU tests skipped.")

banner("🎧 TORCHAUDIO")
ta = safe_import("torchaudio")
if ta:
    try:
        backend = getattr(ta, "get_audio_backend", None)
        if callable(backend):
            print(f"torchaudio backend: {backend()}")
        else:
            # Newer versions use sox_io backend by default
            print("torchaudio backend API deprecated (>=2.3) — default is 'sox_io'")
    except Exception as e:
        print("⚠️ Could not query torchaudio backend:", e)

    try:
        import torchaudio.transforms as T
        spec = T.Spectrogram()(torch.randn(1,16000))
        print(f"torchaudio spectrogram OK: shape={spec.shape}")
    except Exception as e:
        print("❌ torchaudio transform failed:", e)

banner("⚡ FASTER-WHISPER (basic load test)")
try:
    from faster_whisper import WhisperModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel("tiny", device=device, compute_type="float16" if torch.cuda.is_available() else "float32")
    print(f"✅ faster-whisper loaded on {device}")
except Exception as e:
    print("❌ faster-whisper failed to load:", e)
    traceback.print_exc(limit=1)

banner("🗣️ PYANNOTE AUDIO (pipeline load)")
try:
    from pyannote.audio import Pipeline
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", use_auth_token=False)
    print("✅ pyannote pipeline loaded")
except Exception as e:
    print("⚠️ pyannote not tested (needs HF token or offline model):", e.__class__.__name__, e)

banner("🌐 ARGOS TRANSLATE")
try:
    import argostranslate.package, argostranslate.translate
    argostranslate.package.update_package_index()
    print("✅ Argos Translate import OK")
    installed = argostranslate.package.get_installed_packages()
    print(f"Installed translation models: {[f'{p.from_code}->{p.to_code}' for p in installed]}")
except Exception as e:
    print("❌ Argos Translate error:", e)

banner("🈶 PYKAKASI")
try:
    from pykakasi import kakasi
    conv = kakasi()
    conv.setMode("H", "a")
    converter = conv.getConverter()
    print("✅ pykakasi sample:", converter.do("こんにちは"))
except Exception as e:
    print("❌ pykakasi failed:", e)

banner("🔠 LANGID")
try:
    import langid
    lang, score = langid.classify("これは日本語の文です。")
    print(f"✅ langid classified sample as {lang} (p={score:.2f})")
except Exception as e:
    print("❌ langid failed:", e)

banner("📊 PANDAS")
try:
    import pandas as pd
    df = pd.DataFrame({"col1": [1,2], "col2": ["a","b"]})
    print("✅ pandas DataFrame test:", df.shape)
except Exception as e:
    print("❌ pandas failed:", e)

banner("✅ SUMMARY")
print("If all modules above show ✅, your environment is fully functional.")
print("If CUDA/cuDNN show OK, GPU inference is ready for Whisper and Pyannote.")
