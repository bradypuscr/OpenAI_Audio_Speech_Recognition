"""
Quick dependency check for Whisper + Pyannote + TorchCodec + CUDA + Torchaudio

Usage:
    python test_audio_env.py --hf-token <YOUR_HUGGINGFACE_TOKEN>
"""

import importlib
import os
import sys
import math
import argparse
from pathlib import Path


# ---------------------------------------------------------
# CLI ARGUMENTS
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="Audio environment sanity checker")
parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face access token")
args = parser.parse_args()
hf_token = args.hf_token


# ---------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------
def check_pkg(name):
    """Try importing a package and print its version."""
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "n/a")
        print(f"[OK] {name:15s} v{ver}")
        return mod
    except Exception as e:
        print(f"[FAIL] {name:15s} -> {type(e).__name__}: {e}")
        return None


print("=== PYTHON & PATH ===")
print(f"Python: {sys.version}")
print(f"PATH   : {os.environ.get('PATH', '')[:120]}...\n")


# ---------------------------------------------------------
# CORE LIBRARIES
# ---------------------------------------------------------
print("=== CORE LIBRARIES ===")
torch = check_pkg("torch")
torchaudio = check_pkg("torchaudio")
pyannote = check_pkg("pyannote.audio")
fwhisper = check_pkg("faster_whisper")
langid = check_pkg("langid")
pykakasi = check_pkg("pykakasi")
# torchcodec = check_pkg("torchcodec")  # optional


# ---------------------------------------------------------
# TORCH / CUDA
# ---------------------------------------------------------
print("\n=== TORCH / CUDA TEST ===")
if torch:
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version  :", getattr(torch.version, "cuda", "n/a"))
    print("cuDNN version :", torch.backends.cudnn.version())
    x = torch.rand(2, 2).to("cuda" if torch.cuda.is_available() else "cpu")
    print("Tensor device :", x.device)


# ---------------------------------------------------------
# TORCHAUDIO TEST (v2.9+ compatible)
# ---------------------------------------------------------
print("\n=== TORCHAUDIO TEST ===")

if torchaudio and torch:
    print("TorchAudio version:", torchaudio.__version__)
    try:
        sample_rate = 16000
        t = torch.linspace(0, 1, sample_rate)
        waveform = 0.1 * torch.sin(2 * math.pi * 440 * t).unsqueeze(0)
        backends = ["soundfile", "sox_io", "ffmpeg"]
        test_file = Path("test_backend.wav")

        print("\n[Backend Save/Load Test]")
        for backend in backends:
            try:
                torchaudio.save(test_file, waveform, sample_rate, backend=backend)
                loaded, sr = torchaudio.load(test_file, backend=backend)
                print(f"✅ {backend:9s}: OK — loaded {loaded.shape[1]} samples @ {sr} Hz")
                test_file.unlink(missing_ok=True)
            except Exception as e:
                print(f"❌ {backend:9s}: {e}")
        print("✔️  Torchaudio sanity check done.")
    except Exception as e:
        print("⚠️  Torchaudio I/O test failed:", e)


# ---------------------------------------------------------
# TORCHCODEC TEST (optional)
# ---------------------------------------------------------
# print("\n=== TORCHCODEC TEST ===")
# try:
#     import torchcodec
#     _ = dir(torchcodec)
#     print("TorchCodec import successful.")
# except Exception as e:
#     print("TorchCodec load error:", e)


# ---------------------------------------------------------
# PYANNOTE PIPELINE TEST
# ---------------------------------------------------------
print("\n=== PYANNOTE PIPELINE TEST ===")
try:
    from pyannote.audio import Pipeline

    if not hf_token:
        print("⚠️  No --hf-token provided — skipping model load.")
    else:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
        print("Pyannote diarization pipeline loaded OK.")
except Exception as e:
    print("Pyannote test error:", e)


# ---------------------------------------------------------
# FASTER WHISPER TEST
# ---------------------------------------------------------
print("\n=== FASTER WHISPER TEST ===")
try:
    from faster_whisper import WhisperModel

    model = WhisperModel(
        "tiny",
        device="cuda" if torch and torch.cuda.is_available() else "cpu",
        compute_type="float16",
    )
    print("Whisper model loaded OK.")
except Exception as e:
    print("Whisper test error:", e)


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------
print("\n=== SUMMARY ===")
print("If all sections above show [OK] and no exceptions, your environment is ready ✅")
