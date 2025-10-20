from faster_whisper import WhisperModel
from tqdm import tqdm
import sys, os

# Initialize the model (using GPU with your RTX 4080)
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

for path in sys.argv[1:]:
    print(f"🎧 Processing: {path}")

    # Transcribe the audio
    segments, info = model.transcribe(
        path,
        vad_filter=True,
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=True,
    )

    base = os.path.splitext(path)[0]
    out_path = base + "_transcription.txt"

    # Count total segments for progress bar
    segments = list(segments)
    total_segments = len(segments)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Detected primary language: {info.language}\n")
        f.write("=== TRANSCRIPTION ===\n\n")

        # tqdm progress bar
        for s in tqdm(segments, total=total_segments, desc="📝 Transcribing", ncols=80):
            start = s.start
            end = s.end
            f.write(f"[{start:.1f}-{end:.1f}s] {s.text.strip()}\n")

    print(f"✅ Transcription saved to: {out_path}")
