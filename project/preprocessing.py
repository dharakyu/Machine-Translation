# preprocessing
from io import open
import unicodedata
import string
import re

# recycled from other tasks
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

lines = open('data/es-en (1)/europarl-v7.es-en.en', encoding='utf-8').\
        read().strip('\n')
lines = lines.replace('\n', '').split('.')

print(len(lines))
print(lines[:10])

cleaned = []
for i in range(1000):
	cleaned.append(normalizeString(lines[i]))
print(len(cleaned))
print(cleaned[:10])