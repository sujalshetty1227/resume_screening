"""
Text normalisation shared by every stage of the pipeline.

Everything downstream (TF-IDF, skill matching, regex extraction) runs on the
output of `normalize`, so tokenisation rules live in exactly one place.
"""
from __future__ import annotations

import re
from typing import Iterable, List

# Kept deliberately small and hand-audited. A large stopword list starts
# deleting signal ("c" from "c++", "r" from "R") which matters for resumes.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "with", "as", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "it", "its", "this", "that", "these", "those", "we", "our", "you",
    "your", "they", "their", "i", "me", "my", "he", "she", "his", "her", "will",
    "would", "should", "can", "could", "have", "has", "had", "do", "does", "did",
    "not", "no", "so", "than", "then", "there", "here", "up", "out", "about",
    "into", "over", "also", "who", "which", "what", "when", "where", "how",
}

# Characters that carry meaning inside technical tokens and must survive
# tokenisation: c++, c#, node.js, ci/cd, scikit-learn.
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.\-/]*")


def normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip control characters."""
    text = text.replace(" ", " ")
    text = re.sub(r"[^\S\n]+", " ", text.lower())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str, drop_stopwords: bool = True) -> List[str]:
    """Normalised text -> list of tokens.

    Trailing punctuation is stripped ("python." -> "python") but internal
    punctuation is preserved ("node.js" stays intact).
    """
    tokens = _TOKEN_RE.findall(normalize(text))
    tokens = [t.strip(".-/") for t in tokens]
    tokens = [t for t in tokens if t and not t.isdigit() and len(t) <= 40]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def ngrams(tokens: Iterable[str], n: int) -> List[str]:
    """Contiguous n-grams as underscore-joined strings."""
    tokens = list(tokens)
    if n <= 1:
        return tokens
    return ["_".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def featurize(text: str, max_ngram: int = 2) -> List[str]:
    """Tokens plus n-grams up to `max_ngram`.

    Bigrams matter here: "machine learning" and "data pipeline" are single
    concepts, and unigrams alone let a resume mentioning "learning" and
    "machine" separately score as if it had the real skill.
    """
    unigrams = tokenize(text)
    feats: List[str] = list(unigrams)
    for n in range(2, max_ngram + 1):
        feats.extend(ngrams(unigrams, n))
    return feats
