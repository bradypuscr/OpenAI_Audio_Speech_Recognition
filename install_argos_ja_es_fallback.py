#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Instala automáticamente paquetes de Argos Translate para traducción japonés → español.
Si ja→es directo no existe, instala ja→en y en→es como fallback (doble paso offline).
"""

import argostranslate.package, argostranslate.translate
import urllib.request, tempfile, json, os, sys

CATALOG_URL = "https://raw.githubusercontent.com/argosopentech/argospm-index/master/index.json"

def download_catalog():
    print("📥 Descargando catálogo de modelos...")
    catalog_path = os.path.join(tempfile.gettempdir(), "argos_index.json")
    urllib.request.urlretrieve(CATALOG_URL, catalog_path)
    with open(catalog_path, "r", encoding="utf-8") as f:
        return json.load(f)

def install_pair(index, src, tgt):
    pkg = next((p for p in index if p.get("from_code") == src and p.get("to_code") == tgt), None)
    if not pkg:
        print(f"❌ No existe paquete {src}→{tgt} en el catálogo.")
        return False

    # Verificar si trae lista de paquetes
    pkgs_list = pkg.get("packages") or []
    if not pkgs_list:
        print(f"⚠️  El par {src}→{tgt} aparece en el catálogo pero no tiene archivos de modelo disponibles.")
        return False

    # Tomar la primera URL válida
    url = pkgs_list[0].get("url")
    if not url:
        print(f"⚠️  No se encontró URL para {src}→{tgt}.")
        return False

    filename = os.path.join(tempfile.gettempdir(), f"{src}_{tgt}.argosmodel")

    try:
        print(f"⬇️  Descargando modelo {src}→{tgt} ...")
        urllib.request.urlretrieve(url, filename)
        print(f"📦 Instalando paquete {src}→{tgt} ...")
        argostranslate.package.install_from_path(filename)
        print(f"✅ Instalado {src}→{tgt}")
        return True
    except Exception as e:
        print(f"❌ Error al descargar o instalar {src}→{tgt}: {e}")
        return False


def already_installed(src, tgt):
    langs = argostranslate.translate.get_installed_languages()
    src_lang = next((l for l in langs if l.code == src), None)
    if not src_lang: return False
    return any(l.code == tgt for l in src_lang.translations)

def main():
    index = download_catalog()
    langs = argostranslate.translate.get_installed_languages()
    print("Idiomas actualmente instalados:", [l.code for l in langs])

    if already_installed("ja", "es"):
        print("✅ Traducción directa ja→es ya instalada.")
        return

    # Intentar directo primero
    print("🔍 Buscando paquete ja→es ...")
    if install_pair(index, "ja", "es"):
        print("🎉 Traducción directa ja→es lista.")
        return

    # Fallback: ja→en + en→es
    print("⚙️  Usando fallback doble (ja→en + en→es)...")
    ok1 = already_installed("ja", "en") or install_pair(index, "ja", "en")
    ok2 = already_installed("en", "es") or install_pair(index, "en", "es")

    if ok1 and ok2:
        print("✅ Fallback instalado correctamente (ja→en→es).")
    else:
        print("❌ No fue posible instalar los modelos necesarios.")

if __name__ == "__main__":
    main()
