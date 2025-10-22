import argostranslate.package, argostranslate.translate, urllib.request, tempfile, json, os

print("Descargando catálogo de modelos...")
catalog_url = "https://raw.githubusercontent.com/argosopentech/argospm-index/master/index.json"
catalog_path = os.path.join(tempfile.gettempdir(), "argos_index.json")
urllib.request.urlretrieve(catalog_url, catalog_path)

with open(catalog_path, "r", encoding="utf-8") as f:
    index = json.load(f)

pkg = next((p for p in index if p["from_code"] == "ja" and p["to_code"] == "es"), None)
if not pkg:
    raise SystemExit("❌ No se encontró paquete ja→es en el catálogo.")

url = pkg["packages"][0]["url"]
pkg_path = os.path.join(tempfile.gettempdir(), "ja_es.argosmodel")

print(f"Descargando modelo desde:\n{url}")
urllib.request.urlretrieve(url, pkg_path)

print("Instalando paquete de idioma ja→es...")
argostranslate.package.install_from_path(pkg_path)

print("✅ Instalación completada. Traducción japonés → español disponible.")
