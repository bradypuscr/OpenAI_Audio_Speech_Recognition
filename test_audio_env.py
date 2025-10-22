"""
Quick dependency check for Whisper + Pyannote + TorchCodec + CUDA + Torchaudio

Run:
    python test_audio_env.py
"""

import importlib, os, sys

token = os.getenv("HUGGINGFACE_TOKEN")

def check_pkg(name):
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

print("=== CORE LIBRARIES ===")
torch = check_pkg("torch")
torchaudio = check_pkg("torchaudio")
pyannote = check_pkg("pyannote.audio")
# torchcodec = check_pkg("torchcodec")
fwhisper = check_pkg("faster_whisper")
langid = check_pkg("langid")
pykakasi = check_pkg("pykakasi")

print("\n=== TORCH / CUDA TEST ===")
if torch:
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version  :", getattr(torch.version, "cuda", "n/a"))
    print("cuDNN version :", torch.backends.cudnn.version())
    x = torch.rand(2, 2).to("cuda" if torch.cuda.is_available() else "cpu")
    print("Tensor device :", x.device)

print("\n=== TORCHAUDIO TEST ===")
if torchaudio:
    info = torchaudio.list_audio_backends()
    print("Available audio backends:", info)

# print("\n=== TORCHCODEC TEST ===")
# try:
#     import torchcodec
#     _ = dir(torchcodec)
#     print("TorchCodec import successful.")
# except Exception as e:
#     print("TorchCodec load error:", e)

print("\n=== PYANNOTE PIPELINE TEST ===")
try:
    from pyannote.audio import Pipeline
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("⚠️  HUGGINGFACE_TOKEN not set — skipping model load.")
    else:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token= token)
        print("Pyannote diarization pipeline loaded OK.")
except Exception as e:
    print("Pyannote test error:", e)

print("\n=== FASTER WHISPER TEST ===")
try:
    from faster_whisper import WhisperModel
    model = WhisperModel("tiny", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")
    print("Whisper model loaded OK.")
except Exception as e:
    print("Whisper test error:", e)

print("\n=== SUMMARY ===")
print("If all sections above show [OK] and no exceptions, your environment is ready ✅")


# $before = $env:Path
# $env:HUGGINGFACE_TOKEN = "token here"
# $after  = $env:Path
# $before -eq $after   # should print True