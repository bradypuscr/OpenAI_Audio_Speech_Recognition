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
import warnings
import glob
import ctypes
from pathlib import Path

warnings.filterwarnings("ignore", message="The 'backend' parameter is not used by TorchCodec")

# ---------------------------------------------------------
# CLI ARGUMENTS
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="Audio environment sanity checker")
parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face access token")
args = parser.parse_args()
token = args.hf_token


# ---------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------
def print_div(title):
    print(f"\n=== {title} ===")


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
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version  :", getattr(torch.version, "cuda", "n/a"))
    print("cuDNN version :", torch.backends.cudnn.version())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        x = torch.rand(2000, 2000, device=device)
        y = x @ x
        if device == "cuda":
            torch.cuda.synchronize()
        print("✅ Tensor matmul successful on", device)
        cuda_results["torch"] = device == "cuda"
    except Exception as e:
        print("❌ Torch CUDA test failed:", e)
        cuda_results["torch"] = False


# ---------------------------------------------------------
# CUDNN LIBRARY TEST
# ---------------------------------------------------------
print_div("CUDNN LIBRARY TEST")

try:
    import torch.backends.cudnn as cudnn

    cudnn_enabled = cudnn.is_available()
    print("cuDNN available:", cudnn_enabled)
    print("cuDNN version  :", cudnn.version())

    # Attempt to locate libcudnn libraries
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    cudnn_libs = (
        glob.glob("/usr/lib/x86_64-linux-gnu/libcudnn*.so*")
        + glob.glob(f"{conda_prefix}/lib/libcudnn*.so*")
        + glob.glob("/usr/local/cuda/lib64/libcudnn*.so*")
    )

    if cudnn_libs:
        print(f"Found cuDNN libraries: {cudnn_libs[:2]}{'...' if len(cudnn_libs)>2 else ''}")
        try:
            _ = ctypes.CDLL(cudnn_libs[0])
            print("✅ cuDNN shared library successfully loaded.")
        except Exception as e:
            print("⚠️ cuDNN library found but failed to load:", e)
    else:
        print("⚠️ cuDNN library not found in standard paths.")

    # Tiny conv2d test to verify GPU execution
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1).to(device)
        inp = torch.randn(1, 3, 64, 64, device=device)
        with torch.backends.cudnn.flags(enabled=True):
            out = model(inp)
        print(f"✅ cuDNN convolution test OK (output shape {out.shape})")
        cuda_results["cudnn"] = True
    else:
        print("⚠️ CUDA not available; skipping cuDNN GPU test.")
        cuda_results["cudnn"] = False

except Exception as e:
    print("❌ cuDNN test failed:", e)
    cuda_results["cudnn"] = False


# ---------------------------------------------------------
# CUDNN SYMBOL INTEGRITY TEST
# ---------------------------------------------------------
print_div("CUDNN SYMBOL INTEGRITY TEST")

def test_cudnn_symbols():
    possible_libs = []
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    search_paths = [
        "/usr/lib/x86_64-linux-gnu/",
        f"{conda_prefix}/lib/",
        "/usr/local/cuda/lib64/",
    ]
    for path in search_paths:
        possible_libs += glob.glob(os.path.join(path, "libcudnn_ops.so*"))
        possible_libs += glob.glob(os.path.join(path, "libcudnn.so*"))

    if not possible_libs:
        print("⚠️ No cuDNN .so libraries found in expected locations.")
        return False

    for lib_path in possible_libs:
        try:
            cudnn = ctypes.CDLL(lib_path)
            print(f"✅ Loaded: {lib_path}")
            for sym in ["cudnnCreateTensorDescriptor", "cudnnGetVersion"]:
                try:
                    getattr(cudnn, sym)
                    print(f"   ✔ Symbol {sym} found")
                except AttributeError:
                    print(f"   ❌ Missing symbol {sym} in {lib_path}")
                    return False
            print("✅ cuDNN library passed symbol integrity test.")
            return True
        except OSError as e:
            print(f"❌ Failed to load {lib_path}: {e}")

    print("❌ All cuDNN libraries failed to load correctly.")
    return False

cuda_results["cudnn_symbols"] = test_cudnn_symbols()


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

    if not token:
        print("⚠️ No --hf-token provided — skipping model load.")
    else:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
        device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        pipeline.to(torch.device(device))
        try:
            first_param = next(pipeline.parameters())
            print("✅ PyAnnote pipeline loaded on:", first_param.device)
        except Exception:
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
    print(f"{k:12s}: {status}")

if all(cuda_results.values()):
    print("\n🎉 All components successfully used CUDA and cuDNN!")
else:
    print("\n⚠️ Some components ran on CPU or have library issues. Check details above.")
