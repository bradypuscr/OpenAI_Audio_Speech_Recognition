#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os, sys, math, re
from tqdm import tqdm
from faster_whisper import WhisperModel

# Transliteración
try:
    from pykakasi import kakasi
    _kakasi = kakasi()
    _kakasi.setMode("J", "H")  # Kanji -> Hiragana
    _kakasi.setMode("K", "H")  # Katakana -> Hiragana
    _kakasi_hira = _kakasi.getConverter()

    _kakasi2 = kakasi()
    _kakasi2.setMode("J", "a") # Kanji -> romaji
    _kakasi2.setMode("K", "a") # Katakana -> romaji
    _kakasi2.setMode("H", "a") # Hiragana -> romaji
    _kakasi_romaji = _kakasi2.getConverter()
except Exception as e:
    _kakasi = _kakasi_hira = _kakasi2 = _kakasi_romaji = None

# Traducción offline con Argos (opcional)
def load_argos_ja_es():
    try:
        import argostranslate.translate as AT
        pairs = AT.get_installed_languages()
        ja = next((l for l in pairs if l.code == "ja"), None)
        es = next((l for l in pairs if l.code == "es"), None)
        if ja and es:
            return ja.get_translation(es)
    except Exception:
        pass
    return None

_ARGOS_TRANS = load_argos_ja_es()

JA_REGEX = re.compile(r"[\u3040-\u30FF\u4E00-\u9FFF]")  # Hiragana/Katakana/Kanji

def is_japanese(text: str) -> bool:
    if not text: return False
    return bool(JA_REGEX.search(text))

def to_hiragana(text: str) -> str:
    if _kakasi_hira:
        return _kakasi_hira.do(text)
    return text  # fallback

def to_romaji(text: str) -> str:
    if _kakasi_romaji:
        return _kakasi_romaji.do(text)
    return text  # fallback

def translate_ja_to_es(text: str, whisper_model: WhisperModel, audio_path: str, seg_start: float, seg_end: float):
    """
    Intenta Argos ja->es. Si no hay, como fallback corta el audio del segmento con
    parámetros start/end del propio segment (Whisper no expone fácil el recorte aquí),
    así que por simplicidad usamos el propio texto como origen:
    1) Whisper translate no aplica a texto, por lo que si no hay Argos, devolvemos None.
    2) Si hay Argos en->es y quisieras doble salto, necesitarías primero ja->en (no offline aquí).
    """
    if _ARGOS_TRANS:
        try:
            return _ARGOS_TRANS.translate(text)
        except Exception:
            return None
    return None  # sin Argos, dejamos traducción vacía (o podrías enganchar un traductor externo)

def format_srt_timestamp(t):
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = int(t % 60)
    millis = int(round((t - int(t)) * 1000))
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

def write_srt(entries, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (start, end, lines) in enumerate(entries, 1):
            f.write(f"{i}\n{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n")
            for line in lines:
                f.write(line.strip() + "\n")
            f.write("\n")

def main():
    ap = argparse.ArgumentParser(description="SRT de estudio: japonés -> (hiragana + romaji + ES)")
    ap.add_argument("audio", help="Ruta del archivo de audio/video")
    ap.add_argument("-m", "--model", default="largest", help="Whisper: large-v3 recomendado; 'largest' usa alias large-v3 si está disponible")
    ap.add_argument("--output", "-o", default=None, help="SRT de salida (por defecto <audio>_ja_es.srt)")
    ap.add_argument("--compute-type", default="float16", help="float16 | int8_float16 | int8 | float32")
    ap.add_argument("--beam-size", type=int, default=5)
    args = ap.parse_args()

    model_name = "large-v3" if args.model.lower() in ("largest","large","large-v3") else args.model
    base = os.path.splitext(os.path.basename(args.audio))[0]
    out_srt = args.output or f"{base}_ja_es.srt"

    print(f"[INFO] Cargando modelo '{model_name}' en CUDA con compute_type={args.compute_type} ...")
    model = WhisperModel(model_name, device="cuda", compute_type=args.compute_type)

    print("[INFO] Transcribiendo (VAD activado, code-switch safe)...")
    segments, info = model.transcribe(
        args.audio,
        language=None,
        task="transcribe",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        condition_on_previous_text=False,
        beam_size=args.beam_size,
        best_of=max(1, args.beam_size),
        temperature=[0.0, 0.2, 0.4],
        word_timestamps=False
    )

    entries = []
    for s in tqdm(segments, desc="Segmentos JA"):
        text = (s.text or "").strip()
        if not is_japanese(text):
            continue
        hira = to_hiragana(text)
        roma = to_romaji(text)
        es = translate_ja_to_es(text, model, args.audio, s.start, s.end)
        lines = [hira, f"({roma})"]
        if es:
            lines.append(f"— [ES] {es}")
        entries.append((s.start, s.end, lines))

    if not entries:
        print("[AVISO] No se detectó japonés. Se crea SRT vacío.")
    write_srt(entries, out_srt)
    print(f"[OK] SRT de estudio creado: {out_srt}")

if __name__ == "__main__":
    main()
