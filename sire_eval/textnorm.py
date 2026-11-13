"""Transcript normalization and scoring helpers shared by local reports and Foundry graders.

This module is embedded verbatim into the Foundry code-based evaluators (see
``sire_eval/evaluators.py``), so it must stay self-contained: standard library only,
no relative imports, no I/O.

The normalizer mirrors the reading rules used to label the regression corpus:
numbers are canonicalized to digits ("thirty" == "30"), "Dr." == "doctor",
spelled-out letters are joined, and punctuation, fillers, contractions, spacing and
"12:00" vs "12" are ignored.
"""

import re

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_ORDINALS = {
    "1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth", "5th": "fifth",
    "6th": "sixth", "7th": "seventh", "8th": "eighth", "9th": "ninth", "10th": "tenth",
    "11th": "eleventh", "12th": "twelfth",
}
_CONTRACTIONS = {
    "i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are",
    "it's": "it is", "that's": "that is", "what's": "what is", "who's": "who is",
    "where's": "where is", "i'll": "i will", "you'll": "you will", "we'll": "we will",
    "i've": "i have", "i'd": "i would", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "can't": "cannot", "won't": "will not", "isn't": "is not",
    "aren't": "are not", "let's": "let us",
}
_FILLERS = {"uh", "um", "uhm", "umm", "er", "erm", "ah", "hmm", "mm", "mhm"}
# "eigh" is how the corpus TTS scripts spell the letter A; recognizers never emit it.
_SYNONYMS = {"ok": "okay", "dr": "doctor", "alright": "all right", "oclock": "", "eigh": "a"}


def _number_run(tokens, i):
    """Parse a run of number words starting at ``i``. Returns (value, next_index)."""
    n = len(tokens)
    total = 0
    group = 0
    seen = False
    last = None
    while i < n:
        t = tokens[i]
        if t in _TENS:
            if last in ("tens", "unit", "teen"):
                break
            group += _TENS[t]
            last = "tens"
        elif t in _UNITS:
            value = _UNITS[t]
            if last == "unit" or last == "teen":
                break
            if last == "tens" and value >= 10:
                break
            group += value
            last = "teen" if value >= 10 else "unit"
        elif t == "hundred" and seen and last != "hundred":
            group *= 100
            total += group
            group = 0
            last = "hundred"
        elif t == "and" and last == "hundred" and i + 1 < n and (tokens[i + 1] in _UNITS or tokens[i + 1] in _TENS):
            pass
        else:
            break
        seen = True
        i += 1
    return total + group, i


def _words_to_numbers(tokens):
    """Convert runs of number words to digits ("twenty five hundred and eighty" -> "2580")."""
    out = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok in _UNITS or tok in _TENS:
            value, i = _number_run(tokens, i)
            out.append(str(value))
            continue
        if tok == "oh" and out and out[-1].isdigit():
            out.append("0")
            i += 1
            continue
        out.append(tok)
        i += 1
    return _apply_multipliers(out)


def _apply_multipliers(tokens):
    """Fold digit + "thousand"/"hundred" forms ("2 thousand 9" -> "2009", "25 hundred 80" -> "2580")."""
    out = []
    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]
        if t.isdigit() and i + 1 < n and tokens[i + 1] in ("thousand", "hundred"):
            scale = 1000 if tokens[i + 1] == "thousand" else 100
            value = int(t) * scale
            i += 2
            if i < n and tokens[i] == "and" and i + 1 < n and tokens[i + 1].isdigit():
                i += 1
            if i < n and tokens[i].isdigit() and int(tokens[i]) < scale:
                value += int(tokens[i])
                i += 1
            out.append(str(value))
            continue
        out.append(t)
        i += 1
    return out


def _join_runs(tokens):
    """Join runs of single letters ("p c n a" -> "pcna") and adjacent digit groups ("4 1 7" -> "417")."""
    out = []
    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]
        if t.isdigit():
            j = i
            while j < n and tokens[j].isdigit():
                j += 1
            out.append("".join(tokens[i:j]))
            i = j
            continue
        if len(t) == 1 and t.isalpha():
            j = i
            while j < n and len(tokens[j]) == 1 and tokens[j].isalpha():
                j += 1
            out.append("".join(tokens[i:j]))
            i = j
            continue
        out.append(t)
        i += 1
    return out


def normalize_tokens(text):
    """Return the canonical token list used for WER and entity matching."""
    if not text:
        return []
    s = str(text).lower()
    s = s.replace("\u2019", "'").replace("&", " and ")
    s = re.sub(r"\bdr\.", "doctor ", s)
    s = re.sub(r"(\d+):00\b", r"\1", s)
    s = re.sub(r"(\d+):(\d+)", r"\1 \2", s)
    s = re.sub(r"\bo'clock\b", " ", s)
    for k, v in _CONTRACTIONS.items():
        s = re.sub(r"\b" + re.escape(k) + r"\b", v, s)
    s = re.sub(r"[-_/\\.,!?;:\"()\[\]{}<>`*#@|~^=+]", " ", s)
    s = s.replace("'", "")
    s = re.sub(r"\b(\d+)\s?(st|nd|rd|th)\b", lambda m: _ORDINALS.get(m.group(1) + m.group(2), m.group(0)), s)
    s = re.sub(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", " ", s)
    tokens = []
    for t in s.split():
        t = _SYNONYMS.get(t, t)
        if t and t not in _FILLERS:
            tokens.extend(t.split())
    tokens = _words_to_numbers(tokens)
    tokens = _join_runs(tokens)
    return [t for t in tokens if t]


def normalize_text(text):
    return " ".join(normalize_tokens(text))


def compact(text):
    return "".join(normalize_tokens(text))


def edit_distance(ref, hyp):
    """Levenshtein distance between two token (or character) sequences."""
    prev = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        cur = [i] + [0] * len(hyp)
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(hyp)]


def transcript_exact_match(reference, hypothesis):
    """True when the normalized transcripts match, ignoring spacing ("otto line" == "ottoline")."""
    return compact(reference) == compact(hypothesis)


def word_error_rate(reference, hypothesis):
    ref = normalize_tokens(reference)
    hyp = normalize_tokens(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / float(len(ref))


def char_error_rate(reference, hypothesis):
    ref = list(compact(reference))
    hyp = list(compact(hypothesis))
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / float(len(ref))


def soundex(word):
    word = "".join(c for c in word.lower() if c.isalpha())
    if not word:
        return ""
    codes = {}
    for letters, digit in (("bfpv", "1"), ("cgjkqsxz", "2"), ("dt", "3"), ("l", "4"), ("mn", "5"), ("r", "6")):
        for ch in letters:
            codes[ch] = digit
    out = []
    last = codes.get(word[0], "")
    for ch in word[1:]:
        code = codes.get(ch, "")
        if code and code != last:
            out.append(code)
        if ch not in "hw":
            last = code
    return (word[0].upper() + "".join(out) + "000")[:4]


def _span_matches(tokens, target):
    """True when consecutive whole tokens concatenate to ``target`` ("ottoline" + "brackwater" == "ottolinebrackwater")."""
    if not target:
        return False
    for i in range(len(tokens)):
        acc = ""
        for j in range(i, len(tokens)):
            acc += tokens[j]
            if acc == target:
                return True
            if len(acc) >= len(target):
                break
    return False


def entity_word_recall(mentions, hypothesis, soundalike=True):
    """Fraction of entity-mention words present in the hypothesis transcript.

    ``mentions`` lists the entity surface forms as spoken (e.g. "zephyrine in radiology ct").
    A mention whose words, concatenated, equal a run of whole hypothesis words counts as fully
    heard, so spacing variants ("otto line" vs "ottoline") are not penalized. With
    ``soundalike`` the remaining words may match a same-sounding hypothesis word (Soundex),
    mirroring the corpus rule that "Brackwater" heard as "Brakwater" is not a mishear.
    """
    hyp_tokens = normalize_tokens(hypothesis)
    total = 0
    found = 0
    for mention in mentions or []:
        m_tokens = normalize_tokens(mention)
        if not m_tokens:
            continue
        total += len(m_tokens)
        if _span_matches(hyp_tokens, "".join(m_tokens)):
            found += len(m_tokens)
            continue
        pool = list(hyp_tokens)
        for tok in m_tokens:
            if tok in pool:
                pool.remove(tok)
                found += 1
                continue
            if soundalike and tok.isalpha() and len(tok) >= 3:
                code = soundex(tok)
                match = next((h for h in pool if h.isalpha() and len(h) >= 3 and soundex(h) == code), None)
                if match is not None:
                    pool.remove(match)
                    found += 1
    if total == 0:
        return 1.0
    return found / float(total)
