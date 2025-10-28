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

Behavior:
- Verifies torch/CUDA/cuDNN
- Checks cuDNN shared libs & key symbols
- Tests torchaudio I/O & GPU transform
- (Optional) Loads pyannote pipeline and faster-whisper tiny on chosen device
- Preflight checker can *offer* to create a conda activate hook that prepends
  the bundled cuDNN path to LD_LIBRARY_PATH (Linux/WSL) or PATH (Windows)
"""

import importlib
import os
import sys
import math
import argparse
import warnings
import glob
import ctypes
import platform
from pathlib import Path

warnings.filterwarnings("ignore", message="The 'backend' parameter is not used by TorchCodec")

# ---------------------------------------------------------
# CLI ARGUMENTS
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="Audio environment sanity checker")
parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face access token")
parser.add_argument("--check-cudnn-path", action="store_true",
                    help="Run cuDNN path checker first and offer a persistent fix (activation hook).")
args = parser.parse_args()
token = args.hf_token

IS_WINDOWS = platform.system().lower().startswith("win")

# ---------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------
def print_div(title: str):
    print(f"\n=== {title} ===")

def check_pkg(name):
    """Try importing a package and print its version."""
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "n/a")
        print(f"[OK]  {name:15s} v{ver}")
        return mod
    except Exception as e:
        print(f"[FAIL] {name:15s} -> {type(e).__name__}: {e}")
        return None

def conda_env_name_from_prefix(prefix: str) -> str:
    return os.path.basename(prefix.rstrip("/\\"))

def check_and_offer_fix_cudnn_path():
    """
    Detect PyTorch-bundled cuDNN path and ensure it's first in the runtime search path.
    - Linux/WSL: LD_LIBRARY_PATH -> .../site-packages/nvidia/cudnn/lib
    - Windows  : PATH            -> ...\site-packages\nvidia\cudnn\bin
    Offers to create an activation hook under:
      <env>/etc/conda/activate.d/99-cudnn-path.(sh|ps1)
    """
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        print("⚠️ Not inside a conda env; activate your env first.")
        return

    if IS_WINDOWS:
        bundled = os.path.join(conda_prefix, "Lib", "site-packages", "nvidia", "cudnn", "bin")
        exists  = os.path.isdir(bundled)
        path_env = os.environ.get("PATH", "")
        parts = [p for p in path_env.split(";") if p]
        front = parts[0] if parts else None

        print_div("cuDNN PATH CHECK (Windows)")
        print("Conda env     :", conda_prefix)
        print("Bundled cuDNN :", bundled, "| exists:", exists)
        print("PATH (front)  :", front or "(empty)")

        if not exists:
            print("\n❌ Bundled cuDNN directory not found. Reinstall PyTorch wheels (cu126):")
            print("   pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio")
            return

        if front and os.path.normcase(front) == os.path.normcase(bundled):
            print("\n✅ cuDNN path is already first in PATH. Nothing to do.")
            return

        print("\n⚠️ cuDNN path is not first in PATH.")
        print("I can create an activation hook so this env always prepends the correct path.")
        hook_dir = os.path.join(conda_prefix, "etc", "conda", "activate.d")
        hook_ps1 = os.path.join(hook_dir, "99-cudnn-path.ps1")
        contents = f'$env:PATH = "{bundled};" + $env:PATH\n'

        print("\n--- WHAT I WOULD DO ---")
        print(f"1) Create: {hook_ps1}")
        print("   with contents:")
        print("      " + contents.strip())
        print(f"2) Re-activate the env:  conda deactivate && conda activate {conda_env_name_from_prefix(conda_prefix)}")

        ans = input("\nProceed? [y/N]: ").strip().lower()
        if ans != "y":
            print("No changes made. For a temporary fix this session, run:")
            print(f'  set PATH={bundled};%PATH%')
            return

        os.makedirs(hook_dir, exist_ok=True)
        with open(hook_ps1, "w", encoding="utf-8") as f:
            f.write(contents)
        print("\n✅ Hook created.")
        print(f"Now run:  conda deactivate && conda activate {conda_env_name_from_prefix(conda_prefix)}")
        return

    # Linux / WSL
    bundled = os.path.join(conda_prefix, "lib", "python3.12", "site-packages", "nvidia", "cudnn", "lib")
    exists  = os.path.isdir(bundled)
    ld      = os.environ.get("LD_LIBRARY_PATH") or ""
    parts   = [p for p in ld.split(":") if p]
    front   = parts[0] if parts else None

    print_div("cuDNN PATH CHECK (Linux/WSL)")
    print("Conda env     :", conda_prefix)
    print("Bundled cuDNN :", bundled, "| exists:", exists)
    print("LD_LIBRARY_PATH (front):", front or "(empty)")

    if not exists:
        print("\n❌ Bundled cuDNN directory not found. Reinstall PyTorch wheels (cu126):")
        print("   pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio")
        return

    if front == bundled:
        print("\n✅ cuDNN path is already first in LD_LIBRARY_PATH. Nothing to do.")
        return

    print("\n⚠️ cuDNN path is not first in LD_LIBRARY_PATH.")
    print("I can create an activation hook so this env always prepends the correct path.")
    hook_dir = os.path.join(conda_prefix, "etc", "conda", "activate.d")
    hook_sh  = os.path.join(hook_dir, "99-cudnn-path.sh")
    contents = f'export LD_LIBRARY_PATH="{bundled}:$LD_LIBRARY_PATH"\n'

    print("\n--- WHAT I WOULD DO ---")
    print(f"1) Create: {hook_sh}")
    print("   with contents:")
    print("      " + contents.strip())
    print(f"2) Re-activate the env:  conda deactivate && conda activate {conda_env_name_from_prefix(conda_prefix)}")

    ans = input("\nProceed? [y/N]: ").strip().lower()
    if ans != "y":
        print("No changes made. For a temporary fix this session, run:")
        print(f'  export LD_LIBRARY_PATH="{bundled}:$LD_LIBRARY_PATH"')
        return

    os.makedirs(hook_dir, exist_ok=True)
    with open(hook_sh, "w", encoding="utf-8") as f:
        f.write(contents)
    try:
        os.chmod(hook_sh, 0o755)
    except Exception:
        pass

    print("\n✅ Hook created.")
    print(f"Now run:  conda deactivate && conda activate {conda_env_name_from_prefix(conda_prefix)}")

# Optional preflight
if args.check_cudnn_path:
    check_and_offer_fix_cudnn_path()

cuda_results = {}

# ---------------------------------------------------------
# PYTHON & PATH
# ---------------------------------------------------------
print_div("PYTHON & PATH")
print(f"Python: {sys.version}")
cut_path = os.environ.get('PATH', '') if IS_WINDOWS else os.environ.get('PATH', '')
print(f"PATH   : {cut_path[:120]}...\n")

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
        with torch.no_grad():
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

    # Locate libcudnn libraries (include PyTorch-bundled location)
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    bundled_cudnn = os.path.join(
        conda_prefix,
        "Lib" if IS_WINDOWS else "lib",
        "site-packages" if IS_WINDOWS else "python3.12/site-packages",
        "nvidia", "cudnn", "bin" if IS_WINDOWS else "lib"
    )

    cudnn_libs = []
    if IS_WINDOWS:
        # On Windows, DLL names differ; we just verify presence
        if os.path.isdir(bundled_cudnn):
            cudnn_libs.append(bundled_cudnn)
    else:
        cudnn_libs = (
            glob.glob("/usr/lib/x86_64-linux-gnu/libcudnn*.so*")
            + glob.glob(f"{conda_prefix}/lib/libcudnn*.so*")
            + glob.glob("/usr/local/cuda/lib64/libcudnn*.so*")
            + glob.glob(os.path.join(bundled_cudnn, "libcudnn*.so*"))
        )

    if cudnn_libs:
        print(f"Found cuDNN libraries (sample): {cudnn_libs[:2]}{'...' if len(cudnn_libs)>2 else ''}")
        try:
            if not IS_WINDOWS:
                _ = ctypes.CDLL(cudnn_libs[0])
            print("✅ cuDNN shared library presence confirmed.")
        except Exception as e:
            print("⚠️ cuDNN library found but failed to load:", e)
    else:
        print("⚠️ cuDNN library not found in standard paths.")

    # Tiny conv2d test to verify GPU execution
    if torch.cuda.is_available():
        device = torch.device("cuda")
        with torch.no_grad():
            model = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1).to(device)
            inp = torch.randn(1, 3, 64, 64, device=device)
            with torch.backends.cudnn.flags(enabled=True):
                out = model(inp)
            torch.cuda.synchronize()
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
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if IS_WINDOWS:
        # Skip symbol probing on Windows; different API naming & DLL handling
        cudnn_bin = os.path.join(conda_prefix, "Lib", "site-packages", "nvidia", "cudnn", "bin")
        if os.path.isdir(cudnn_bin):
            print(f"✅ Bundled cuDNN bin exists: {cudnn_bin}")
            return True
        print("⚠️ Bundled cuDNN bin not found.")
        return False

    bundled = os.path.join(conda_prefix, "lib", "python3.12", "site-packages", "nvidia", "cudnn", "lib")
    search_paths = [
        "/usr/lib/x86_64-linux-gnu/",
        f"{conda_prefix}/lib/",
        "/usr/local/cuda/lib64/",
        bundled,  # include PyTorch-bundled cuDNN
    ]

    possible_libs = []
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

if 'torchaudio' in sys.modules and torch:
    torchaudio = sys.modules['torchaudio']
    print("TorchAudio version:", torchaudio.__version__)
    try:
        sample_rate = 16000
        t = torch.linspace(0, 1, sample_rate)
        waveform = 0.1 * torch.sin(2 * math.pi * 440 * t).unsqueeze(0)
        test_file = Path("test_backend.wav")

        print("\n[Backend Save/Load Test]")
        # Save with stable backends; ffmpeg often can't save in torchaudio
        for backend in ["soundfile", "sox_io"]:
            try:
                torchaudio.save(test_file, waveform, sample_rate, backend=backend)
                loaded, sr = torchaudio.load(test_file, backend=backend)
                print(f"✅ {backend:9s}: OK — loaded {loaded.shape[1]} samples @ {sr} Hz")
                test_file.unlink(missing_ok=True)
            except Exception as e:
                print(f"❌ {backend:9s}: {e}")

        # Optional: try loading with ffmpeg backend if present
        try:
            loaded, sr = torchaudio.load(test_file, backend="ffmpeg")
            print(f"✅ {'ffmpeg':9s}: load OK — {loaded.shape[1]} samples @ {sr} Hz")
        except Exception as e:
            print(f"ℹ️ {'ffmpeg':9s}: load skipped or unavailable ({e})")

        # GPU Transform test
        print("\n[GPU Transform Test]")
        if torch.cuda.is_available():
            spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=64).to("cuda")
            waveform_gpu = waveform.to("cuda")
            with torch.no_grad():
                out = spec(waveform_gpu)
                torch.cuda.synchronize()
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
        # modern kwarg name:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
        device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        pipeline.to(torch.device(device))
        try:
            first_param = next(pipeline.parameters())
            print("✅ PyAnnote pipeline loaded on:", first_param.device)
        except Exception:
            print(f"✅ PyAnnote pipeline configured for device: {device}")
        cuda_results["pyannote"] = (device == "cuda")
except Exception as e:
    print("PyAnnote test error:", e)

# ---------------------------------------------------------
# FASTER WHISPER TEST
# ---------------------------------------------------------
print_div("FASTER WHISPER TEST")
cuda_results["whisper"] = False
try:
    from faster_whisper import WhisperModel
    use_cuda = bool(torch and torch.cuda.is_available())
    compute_type = "float16" if use_cuda else "int8"  # avoid fp16 on CPU
    device = "cuda" if use_cuda else "cpu"
    model = WhisperModel("tiny", device=device, compute_type=compute_type)
    print(f"✅ Whisper model loaded on {device} (compute_type={compute_type})")
    cuda_results["whisper"] = use_cuda
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
