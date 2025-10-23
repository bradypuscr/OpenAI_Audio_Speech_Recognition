# test_pykakasi.py
from pykakasi import kakasi

# Create instance
kks = kakasi()

# Convert Japanese text to hiragana and romaji
text = "私は学生です。"
kks.setMode("J", "H")  # Kanji → Hiragana
kks.setMode("K", "H")  # Katakana → Hiragana
conv_hira = kks.getConverter()
hiragana = conv_hira.do(text)

kks.setMode("J", "a")  # Kanji → Romaji
kks.setMode("K", "a")  # Katakana → Romaji
kks.setMode("H", "a")  # Hiragana → Romaji
conv_romaji = kks.getConverter()
romaji = conv_romaji.do(text)

print("Original:", text)
print("Hiragana:", hiragana)
print("Romaji:", romaji)
