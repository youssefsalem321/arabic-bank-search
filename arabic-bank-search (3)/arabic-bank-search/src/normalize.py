import re

_TASHKEEL = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u0640]")  # harakat + tatweel
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")


def normalize(text: str) -> str:
    """Light Arabic normalization: diacritics, alef/ya/ta-marbuta variants, digits, punctuation."""
    text = text.translate(_AR_DIGITS)
    text = _TASHKEEL.sub("", text)
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text: str) -> list[str]:
    """Normalize + strip common prefixes (used by BM25 only)."""
    out = []
    for tok in normalize(text).split():
        for p in _PREFIXES:
            if tok.startswith(p) and len(tok) - len(p) >= 3:
                tok = tok[len(p):]
                break
        out.append(tok)
    return out
