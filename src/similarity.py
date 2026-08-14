"""
Vector-space similarity, implemented from scratch on numpy.

Why not scikit-learn: the whole scoring story of this agent lives in this file,
and a from-scratch implementation keeps it inspectable (and drops a ~60MB
transitive dependency). The maths is standard and documented inline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .text_utils import featurize


@dataclass
class TfidfIndex:
    """A fitted TF-IDF vector space.

    Fitted over the JD *and* every resume together, so IDF reflects this
    candidate pool. A term appearing in every resume ("python" for an ML role)
    is correctly discounted toward zero -- it does not discriminate.
    """

    vocabulary: Dict[str, int] = field(default_factory=dict)
    idf: np.ndarray = field(default_factory=lambda: np.zeros(0))
    max_ngram: int = 2

    @classmethod
    def fit(cls, documents: Sequence[str], max_ngram: int = 2) -> "TfidfIndex":
        tokenized = [featurize(doc, max_ngram) for doc in documents]

        # Document frequency: number of documents containing each term.
        df: Dict[str, int] = {}
        for tokens in tokenized:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1

        vocabulary = {term: i for i, term in enumerate(sorted(df))}
        n_docs = len(documents)

        # Smoothed IDF, identical to sklearn's default:
        #     idf(t) = ln((1 + N) / (1 + df(t))) + 1
        # The +1s prevent division by zero and stop a term present in every
        # document from collapsing to exactly 0 (which would erase it).
        idf = np.zeros(len(vocabulary), dtype=np.float64)
        for term, i in vocabulary.items():
            idf[i] = math.log((1 + n_docs) / (1 + df[term])) + 1.0

        return cls(vocabulary=vocabulary, idf=idf, max_ngram=max_ngram)

    def transform(self, document: str) -> np.ndarray:
        """Document -> L2-normalised TF-IDF vector.

        Sublinear TF (1 + log(count)) is used so a resume that repeats
        "python" fifteen times does not out-score one that demonstrates it
        once in context. Keyword stuffing is a real failure mode in resumes.
        """
        vec = np.zeros(len(self.vocabulary), dtype=np.float64)
        counts: Dict[str, int] = {}
        for term in featurize(document, self.max_ngram):
            counts[term] = counts.get(term, 0) + 1

        for term, count in counts.items():
            idx = self.vocabulary.get(term)
            if idx is not None:
                vec[idx] = (1.0 + math.log(count)) * self.idf[idx]

        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of the angle between two vectors.

    Inputs from TfidfIndex.transform are already L2-normalised, so this is a
    dot product; the explicit norms keep the function correct if reused.
    """
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))


class BM25:
    """Okapi BM25 -- an alternative ranking function, exposed via --similarity bm25.

    BM25 differs from TF-IDF cosine in two ways that matter for resumes:
      * term frequency saturates (k1) instead of growing without bound, and
      * document length is normalised (b), so a 4-page resume is not
        automatically penalised against a 1-page one.
    Included so the scoring choice is an evidenced comparison, not an assertion.
    """

    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75,
                 max_ngram: int = 2) -> None:
        self.k1, self.b, self.max_ngram = k1, b, max_ngram
        self.docs: List[List[str]] = [featurize(d, max_ngram) for d in documents]
        self.doc_len = np.array([len(d) for d in self.docs], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if len(self.docs) else 0.0

        self.df: Dict[str, int] = {}
        for tokens in self.docs:
            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1
        self.n_docs = len(self.docs)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        # BM25's probabilistic IDF, +1 inside the log to keep it non-negative.
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def score(self, query: str, doc_index: int) -> float:
        tokens = self.docs[doc_index]
        if not tokens:
            return 0.0
        counts: Dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1

        length_norm = 1 - self.b + self.b * (self.doc_len[doc_index] / (self.avgdl or 1))
        total = 0.0
        for term in set(featurize(query, self.max_ngram)):
            f = counts.get(term, 0)
            if f == 0:
                continue
            total += self._idf(term) * (f * (self.k1 + 1)) / (f + self.k1 * length_norm)
        return total

    def scores(self, query: str) -> np.ndarray:
        return np.array([self.score(query, i) for i in range(self.n_docs)])
