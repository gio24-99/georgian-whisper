
import re
import unicodedata

# Mtavruli (Georgian capital letters, U+1C90-U+1CBF) map 1:1 to Mkhedruli (U+10D0-U+10FF).
_MTAVRULI_START, _MTAVRULI_END = 0x1C90, 0x1CBF
_MTAVRULI_OFFSET = 0x1C90 - 0x10D0

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACES = re.compile(r"\s+")


def normalize_ka(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "".join(
        chr(ord(c) - _MTAVRULI_OFFSET) if _MTAVRULI_START <= ord(c) <= _MTAVRULI_END else c
        for c in text
    )
    text = text.lower()  # only affects Latin characters mixed into Georgian text
    text = _PUNCT.sub(" ", text).replace("_", " ")
    return _SPACES.sub(" ", text).strip()


if __name__ == "__main__":
    print(normalize_ka("გამარჯობა, როგორ ხარ?  ᲒᲐᲛᲐᲠᲯᲝᲑᲐ!"))
