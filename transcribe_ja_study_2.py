#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transcripción bilingüe español/japonés con salida SRT enriquecida.
- Detecta automáticamente idioma por segmento (code-switching friendly)
- Para japonés: agrega hiragana, romaji y (opcionalmente) traducción al español (--translate)
- Para español: conserva texto original
- Soporte CUDA / GPU (float16, int8_float16, etc)
- Verbose mode (--verbose) para diagnósticos y seguimiento detallado
"""

import argparse, os, math, re, time, requests
from tqdm import tqdm
from faster_whisper import WhisperModel

# ========== Transliteración ==========
try:
    from pykakasi import kakasi
    kakasi_h = kakasi(); kakasi_h.setMode("J","H"); kakasi_h.setMode("K","H")
    conv_hira = kakasi_h.getConverter()
    kakasi_r = kakasi(); kakasi_r.setMode("J","a"); kakasi_r.setMode("K","a"); kakasi_r.setMode("H","a")
    conv_romaji = kakasi_r.getConverter()
except Exception:
    conv_hira = conv_romaji = None

# ========== Traducción ==========
def load_argos_ja_es(verbose=False):
    try:
        import argostranslate.translate as AT
        langs = AT.get_installed_languages()
        ja = next((l for l in langs if l.code == "ja"), None)
        es = next((l for l in langs if l.code == "es"), None)
        if ja and es:
            if verbose:
                print("[VERBOSE] Traducción ja→es disponible con Argos Translate.")
            return ja.get_translation(es)
        else:
            if verbose:
                print("[VERBOSE] No se encontró paquete Argos ja→es instalado.")
    except Exception as e:
        if verbose:
            print(f"[VERBOSE] Error cargando Argos: {e}")
    return None

def translate_online(text, source="ja", target="es", verbose=False):
    """Traducción usando LibreTranslate (gratuito, sin autenticación)."""
    url = "https://libretranslate.de/translate"
    try:
        response = requests.post(
            url,
            json={"q": text, "source": source, "target": target, "format": "text"},
            timeout=15
        )
        if response.status_code == 200:
            result = response.json().get("translatedText", "")
            if verbose:
                print(f"[VERBOSE] Traducción online ({source}->{target}) OK.")
            return result
        else:
            if verbose:
                print(f"[VERBOSE] Error HTTP {response.status_code} en traducción online.")
    except Exception as e:
        if verbose:
            print(f"[VERBOSE] Error en traducción online: {e}")
    return ""

argos_trans = None

def translate_ja_es(txt, verbose=False):
    """Usa Argos si está disponible; si no, usa traducción online."""
    if argos_trans:
        try:
            return argos_trans.translate(txt)
        except Exception as e:
            if verbose:
                print(f"[VERBOSE] Error traduciendo con Argos: {e}")
    # fallback
    return translate_online(txt, "ja", "es", verbose)

# ========== Detección idioma ==========
JA_REGEX = re.compile(r"[\u3040-\u30FF\u4E00-\u9FFF]")
def is_japanese(text):
    return bool(JA_REGEX.search(text or ""))

# ========== Utilidades ==========
def fmt_srt(t):
    h=int(t//3600); m=int((t%3600)//60); s=int(t%60); ms=int(round((t-int(t))*1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def next_available_filename(base_path):
    """Si el archivo existe, crea una versión numerada."""
    if not os.path.exists(base_path):
        return base_path
    root, ext = os.path.splitext(base_path)
    n = 1
    while True:
        candidate = f"{root}_{n}{ext}"
        if not os.path.exists(candidate):
            return candidate
        n += 1

def write_srt(entries, out_path):
    with open(out_path,"w",encoding="utf-8") as f:
        for i,(start,end,lines) in enumerate(entries,1):
            f.write(f"{i}\n{fmt_srt(start)} --> {fmt_srt(end)}\n")
            for line in lines:
                f.write(line+"\n")
            f.write("\n")

# ========== MAIN ==========
def main():
    ap = argparse.ArgumentParser(description="Transcripción español+japonés (multi-idioma, SRT con traducción opcional).")
    ap.add_argument("audio", help="Ruta al archivo de audio/video")
    ap.add_argument("--model", default="large-v3", help="Modelo Whisper (ej. large-v3, medium, small, etc.)")
    ap.add_argument("--compute-type", default="float16", help="float16 | int8_float16 | int8 | float32")
    ap.add_argument("--beam-size", type=int, default=5, help="Beam size para decodificación")
    ap.add_argument("--output", "-o", help="Archivo de salida opcional (.srt)")
    ap.add_argument("--translate", action="store_true", help="Traducir japonés al español (por defecto NO traduce)")
    ap.add_argument("-v", "--verbose", action="store_true", help="Modo detallado (muestra información avanzada)")
    args = ap.parse_args()

    if args.verbose:
        print("="*80)
        print("[VERBOSE] Inicio de ejecución detallada")
        print(f"[VERBOSE] Archivo de entrada : {args.audio}")
        print(f"[VERBOSE] Modelo a usar      : {args.model}")
        print(f"[VERBOSE] Compute type       : {args.compute_type}")
        print(f"[VERBOSE] Beam size          : {args.beam_size}")
        print(f"[VERBOSE] CUDA               : activado (por defecto)")
        print(f"[VERBOSE] Traducción         : {'ACTIVADA' if args.translate else 'DESACTIVADA'}")
        print("="*80)

    if not os.path.exists(args.audio):
        print(f"[ERROR] Archivo no encontrado: {args.audio}")
        return

    input_dir = os.path.dirname(os.path.abspath(args.audio))
    base_name = os.path.splitext(os.path.basename(args.audio))[0]
    out_path = args.output or os.path.join(input_dir, f"{base_name}_mix_study.srt")
    out_path = next_available_filename(out_path)

    if args.verbose:
        print(f"[VERBOSE] Directorio de salida : {input_dir}")
        print(f"[VERBOSE] Archivo de salida    : {out_path}")

    # ========= Carga modelo =========
    if args.verbose:
        print("[VERBOSE] Cargando modelo Whisper...")
        start_load = time.time()

    model = WhisperModel(args.model, device="cuda", compute_type=args.compute_type)

    if args.verbose:
        print(f"[VERBOSE] Modelo cargado en {time.time()-start_load:.2f}s")

    global argos_trans
    if args.translate:
        if args.verbose:
            print("[VERBOSE] Intentando carga de Argos Translate...")
        argos_trans = load_argos_ja_es(args.verbose)
    else:
        if args.verbose:
            print("[VERBOSE] Traducción al español desactivada (--translate no usado).")

    # ========= Transcripción =========
    if args.verbose:
        print("[VERBOSE] Iniciando transcripción...")
        print("           - VAD activado (filtrado de silencios)")
        print("           - condition_on_previous_text = False")
        print("           - temperature = [0.0, 0.2, 0.4]")
        print("           - beam_size =", args.beam_size)
        print("-"*80)

    segments, info = model.transcribe(
        args.audio,
        language=None,
        task="transcribe",
        vad_filter=True,
        condition_on_previous_text=False,
        beam_size=args.beam_size,
        best_of=args.beam_size,
        temperature=[0.0, 0.2, 0.4]
    )

    if args.verbose:
        print(f"[VERBOSE] Idioma predominante detectado: {info.language} "
              f"(p={info.language_probability:.2f})")

    entries=[]
    for seg in tqdm(segments, desc="Procesando segmentos"):
        txt=(seg.text or "").strip()
        if not txt:
            continue

        if is_japanese(txt):
            hira = conv_hira.do(txt) if conv_hira else txt
            roma = conv_romaji.do(txt) if conv_romaji else ""
            lines=[txt, hira, f"({roma})"]

            esp = ""
            if args.translate:
                esp = translate_ja_es(txt, args.verbose)
                if esp:
                    lines.append(f"— [ES] {esp}")

            if args.verbose:
                print(f"[VERBOSE] Segmento japonés [{seg.start:.2f}-{seg.end:.2f}] -> {txt[:40]}...")
        else:
            lines=[txt]
            if args.verbose:
                print(f"[VERBOSE] Segmento no-japonés [{seg.start:.2f}-{seg.end:.2f}] -> {txt[:40]}...")
        entries.append((seg.start, seg.end, lines))

    write_srt(entries, out_path)
    print(f"[OK] Archivo creado: {out_path}")

    if args.verbose:
        print(f"[VERBOSE] Total de segmentos procesados: {len(entries)}")
        print(f"[VERBOSE] Ejecución completada correctamente.")
        print("="*80)

if __name__ == "__main__":
    main()
