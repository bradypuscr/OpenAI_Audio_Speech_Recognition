"""
Full environment sanity test for Whisper + Pyannote + TorchCodec + CUDA + Torchaudio


sudo apt update && sudo apt upgrade -y
sudo apt install curl wget git -y
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh
source ~/.bashrc
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r


conda deactivate
conda env remove --name py312 -y
conda clean --all -y

conda create -n py312 python=3.12 -y
conda activate py312

conda install "ffmpeg<8"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install torchcodec --index-url=https://download.pytorch.org/whl/cu126
pip install pyannote.audio





print(torch.cuda.is_available())


sudo apt install -y nvidia-cuda-toolkit
conda uninstall ffmpeg
conda install "ffmpeg<6" -c conda-forge
conda install libiconv -c conda-forge
pip install torchcodec --index-url=https://download.pytorch.org/whl/cu128
pip install pykakasi faster-whisper argostranslate langid



Usage:
    python test_audio_env.py --hf-token <YOUR_HUGGINGFACE_TOKEN>
"""

import importlib
import os
import sys
import math
import argparse
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", message="The 'backend' parameter is not used by TorchCodec")

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


def print_div(title):
    print(f"\n=== {title} ===")


def cuda_status(torch):
    return torch.cuda.is_available(), getattr(torch.version, "cuda", "n/a"), torch.backends.cudnn.version()


cuda_results = {}

# ---------------------------------------------------------
# PYTHON & PATH
# ---------------------------------------------------------
print_div("PYTHON & PATH")
print(f"Python: {sys.version}")
print(f"PATH   : {os.environ.get('PATH', '')[:120]}...\n")


# ---------------------------------------------------------
# CORE LIBRARIES
# ---------------------------------------------------------
print_div("CORE LIBRARIES")
torch = check_pkg("torch")
torchaudio = check_pkg("torchaudio")
pyannote = check_pkg("pyannote.audio")
fwhisper = check_pkg("faster_whisper")
langid = check_pkg("langid")
pykakasi = check_pkg("pykakasi")
# torchcodec = check_pkg("torchcodec")  # optional


# ---------------------------------------------------------
# TORCH / CUDA TEST
# ---------------------------------------------------------
print_div("TORCH / CUDA TEST")
if torch:
    is_cuda, cuda_ver, cudnn_ver = cuda_status(torch)
    print("CUDA available:", is_cuda)
    print("CUDA version  :", cuda_ver)
    print("cuDNN version :", cudnn_ver)

    device = "cuda" if is_cuda else "cpu"
    try:
        x = torch.rand(2000, 2000, device=device)
        y = x @ x
        torch.cuda.synchronize() if is_cuda else None
        print("✅ Tensor matmul successful on", device)
        cuda_results["torch"] = is_cuda
    except Exception as e:
        print("❌ Torch CUDA test failed:", e)
        cuda_results["torch"] = False


# ---------------------------------------------------------
# TORCHAUDIO TEST (I/O + GPU Transform)
# ---------------------------------------------------------
print_div("TORCHAUDIO TEST")

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

        # GPU Transform test
        print("\n[GPU Transform Test]")
        if torch.cuda.is_available():
            spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=64).to("cuda")
            waveform_gpu = waveform.to("cuda")
            out = spec(waveform_gpu)
            print(f"✅ MelSpectrogram executed on device: {out.device}")
            cuda_results["torchaudio"] = True
        else:
            print("⚠️ CUDA not available for Torchaudio transforms.")
            cuda_results["torchaudio"] = False

    except Exception as e:
        print("⚠️ Torchaudio I/O test failed:", e)
        cuda_results["torchaudio"] = False


# ---------------------------------------------------------
# PYANNOTE PIPELINE TEST
# ---------------------------------------------------------
print_div("PYANNOTE PIPELINE TEST")
cuda_results["pyannote"] = False
try:
    from pyannote.audio import Pipeline

    if not hf_token:
        print("⚠️ No --hf-token provided — skipping model load.")
    else:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
        device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        pipeline.to(torch.device(device))
        try:
            # For older pyannote versions
            first_param = next(pipeline.parameters())
            print("✅ PyAnnote pipeline loaded on:", first_param.device)
        except Exception:
            # For newer pyannote >= 3.3 where .parameters() doesn't exist
            print(f"✅ PyAnnote pipeline configured for device: {device}")
        cuda_results["pyannote"] = device == "cuda"

except Exception as e:
    print("PyAnnote test error:", e)


# ---------------------------------------------------------
# FASTER WHISPER TEST
# ---------------------------------------------------------
print_div("FASTER WHISPER TEST")
cuda_results["whisper"] = False
try:
    from faster_whisper import WhisperModel

    device = "cuda" if torch and torch.cuda.is_available() else "cpu"
    model = WhisperModel("tiny", device=device, compute_type="float16")
    print(f"✅ Whisper model loaded on {device}")
    cuda_results["whisper"] = device == "cuda"
except Exception as e:
    print("Whisper test error:", e)


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------
print_div("SUMMARY")

for k, used_cuda in cuda_results.items():
    status = "✅ GPU" if used_cuda else "⚠️ CPU"
    print(f"{k:10s}: {status}")

if all(cuda_results.values()):
    print("\n🎉 All components successfully used CUDA!")
else:
    print("\n⚠️ Some components ran on CPU. Check above for details.")
