import argostranslate.package
import argostranslate.translate

# Define language codes
from_code = "ja"
middle_code = "en"
to_code = "es"

# Update Argos package index
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()

# Helper function to install a specific package
def install_package(from_code, to_code):
    package = next(
        filter(
            lambda x: x.from_code == from_code and x.to_code == to_code,
            available_packages
        ),
        None
    )
    if package:
        print(f"Installing {from_code} → {to_code}")
        argostranslate.package.install_from_path(package.download())
    else:
        print(f"No package found for {from_code} → {to_code}")

# Install both translation directions
install_package("ja", "en")
install_package("en", "es")

# Now translate step by step
text = "いてきます、いてらしゃい。"

# Japanese → English
intermediate = argostranslate.translate.translate(text, "ja", "en")
print("JA → EN:", intermediate)

# English → Spanish
translatedText = argostranslate.translate.translate(intermediate, "en", "es")
print("EN → ES:", translatedText)
