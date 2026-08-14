"""Tests for the from-scratch TF-IDF / BM25 implementation."""
import unittest

from src.similarity import BM25, TfidfIndex, cosine_similarity


class TestTfidf(unittest.TestCase):
    def setUp(self):
        self.docs = [
            "python pytorch nlp transformers docker",
            "python pytorch nlp transformers kubernetes",
            "java spring boot hibernate oracle",
        ]
        self.index = TfidfIndex.fit(self.docs)

    def test_identical_documents_score_one(self):
        vec = self.index.transform(self.docs[0])
        self.assertAlmostEqual(cosine_similarity(vec, vec), 1.0, places=6)

    def test_disjoint_documents_score_zero(self):
        a = self.index.transform(self.docs[0])
        b = self.index.transform(self.docs[2])
        self.assertAlmostEqual(cosine_similarity(a, b), 0.0, places=6)

    def test_similar_beats_dissimilar(self):
        query = self.index.transform(self.docs[0])
        near = cosine_similarity(query, self.index.transform(self.docs[1]))
        far = cosine_similarity(query, self.index.transform(self.docs[2]))
        self.assertGreater(near, far)

    def test_idf_discounts_ubiquitous_terms(self):
        """A term in every document must carry less weight than a rare one."""
        idx = TfidfIndex.fit(["python docker", "python kubernetes", "python aws"])
        common = idx.idf[idx.vocabulary["python"]]
        rare = idx.idf[idx.vocabulary["docker"]]
        self.assertLess(common, rare)

    def test_sublinear_tf_resists_keyword_stuffing(self):
        """Repeating a keyword 20x must not score 20x higher than saying it once."""
        idx = TfidfIndex.fit(["python nlp docker", "java spring"])
        honest = idx.transform("python nlp docker")
        stuffed = idx.transform("python " * 20 + "nlp docker")
        query = idx.transform("python nlp docker")
        gain = cosine_similarity(query, stuffed) - cosine_similarity(query, honest)
        self.assertLess(abs(gain), 0.25)

    def test_empty_document_is_safe(self):
        self.assertEqual(cosine_similarity(self.index.transform(""),
                                           self.index.transform(self.docs[0])), 0.0)

    def test_vectors_are_l2_normalised(self):
        import numpy as np
        vec = self.index.transform(self.docs[0])
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=6)


class TestBM25(unittest.TestCase):
    def test_scores_are_ordered_by_relevance(self):
        docs = ["python pytorch nlp", "python pytorch", "java spring"]
        scores = BM25(docs).scores("python pytorch nlp")
        self.assertEqual(int(scores.argmax()), 0)
        self.assertEqual(int(scores.argmin()), 2)

    def test_term_frequency_saturates(self):
        """BM25's k1 must stop repetition from scaling the score linearly."""
        bm25 = BM25(["python " * 1, "python " * 30, "java spring"])
        once, thirty = bm25.score("python", 0), bm25.score("python", 1)
        self.assertLess(thirty, once * 3)


if __name__ == "__main__":
    unittest.main()
