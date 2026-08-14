"""End-to-end tests over the committed sample corpus (offline mode, no network)."""
import json
import tempfile
import unittest
from pathlib import Path

from src.agent import ResumeScreeningAgent, write_outputs
from src.config import DEFAULT_JD_PATH, MIN_REQUIRED_COVERAGE, RESUME_DIR


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        agent = ResumeScreeningAgent(mode="offline")
        cls.jd = agent.load_job_description(DEFAULT_JD_PATH)
        cls.profiles = agent.load_profiles(RESUME_DIR)
        cls.ranked = agent.rank(cls.jd, cls.profiles)

    def test_processes_more_than_ten_resumes(self):
        self.assertGreaterEqual(len(self.ranked), 10)

    def test_all_three_file_formats_parsed(self):
        suffixes = {Path(p.source_file).suffix for p in self.profiles}
        self.assertTrue({".pdf", ".docx", ".txt"}.issubset(suffixes))

    def test_ranks_are_contiguous_and_sorted_within_each_gate_group(self):
        """Scores descend within a gate group, not necessarily across groups.

        A global `sorted(scores, reverse=True)` assertion is WRONG here and
        contradicts the gate: a gated candidate is deliberately placed below an
        un-gated one with a lower score. That is the whole point of the gate.
        With SEMANTIC_SCALING=raw the sample corpus actually produces such an
        inversion (gated 0.536 below un-gated 0.528), so the global assertion
        was a latent failure the default config happened to hide.
        """
        self.assertEqual([c.rank for c in self.ranked],
                         list(range(1, len(self.ranked) + 1)))
        passed = [c.score.final_score for c in self.ranked
                  if c.score.required_coverage >= MIN_REQUIRED_COVERAGE]
        gated = [c.score.final_score for c in self.ranked
                 if c.score.required_coverage < MIN_REQUIRED_COVERAGE]
        self.assertEqual(passed, sorted(passed, reverse=True))
        self.assertEqual(gated, sorted(gated, reverse=True))

    def test_best_fit_candidate_ranks_first(self):
        """Priya Raghavan is the hand-labelled best match for this JD."""
        self.assertEqual(self.ranked[0].profile.name, "Priya Raghavan")

    def test_off_domain_candidate_is_rejected(self):
        """A 7-year Java backend engineer must not reach the shortlist."""
        rohit = next(c for c in self.ranked if c.profile.name == "Rohit Sharma")
        self.assertTrue(rohit.recommendation.startswith("Reject"))

    def test_hard_gate_vetoes_regardless_of_score(self):
        for c in self.ranked:
            if c.score.required_coverage < MIN_REQUIRED_COVERAGE:
                self.assertTrue(c.recommendation.startswith("Reject"),
                                f"{c.profile.name} passed the gate it should fail")

    def test_unreadable_pdf_is_flagged_not_silently_zeroed(self):
        broken = next(c for c in self.ranked
                      if c.profile.source_file == "scanned_unreadable.pdf")
        self.assertTrue(broken.profile.warnings)
        self.assertIn("scanned", " ".join(broken.profile.warnings).lower())

    def test_run_is_deterministic(self):
        agent = ResumeScreeningAgent(mode="offline")
        again = agent.rank(self.jd, agent.load_profiles(RESUME_DIR))
        self.assertEqual([c.profile.candidate_id for c in again],
                         [c.profile.candidate_id for c in self.ranked])
        self.assertEqual([c.score.final_score for c in again],
                         [c.score.final_score for c in self.ranked])

    def test_every_score_is_explainable(self):
        for c in self.ranked:
            self.assertTrue(c.score.reasoning)
            self.assertAlmostEqual(sum(c.score.weights_used.values()), 1.0, places=6)

    def test_bm25_backend_produces_a_valid_ranking(self):
        agent = ResumeScreeningAgent(mode="offline", similarity="bm25")
        ranked = agent.rank(self.jd, self.profiles)
        self.assertEqual(len(ranked), len(self.ranked))
        self.assertTrue(all(0.0 <= c.score.final_score <= 1.0 for c in ranked))

    def test_gated_candidates_sort_below_everyone_else(self):
        """The gate must move the candidate, not just relabel them."""
        gated = [i for i, c in enumerate(self.ranked)
                 if c.score.required_coverage < MIN_REQUIRED_COVERAGE]
        if gated:
            self.assertEqual(gated, list(range(min(gated), len(self.ranked))),
                             "a gated candidate is ranked above an un-gated one")

    def test_phone_numbers_are_extracted(self):
        """Regression: the old regex missed every +91 number in the corpus."""
        readable = [p for p in self.profiles if p.raw_text.strip()]
        with_phone = [p for p in readable if p.phone]
        self.assertGreaterEqual(len(with_phone), len(readable) - 1)

    def test_education_component_actually_discriminates(self):
        """Regression: the degree regex matched the word "be", so every
        candidate scored 1.0 and the component did no work."""
        levels = {p.degree_level for p in self.profiles}
        self.assertGreater(len(levels), 1, "degree detection is a constant")

    def test_rejects_a_job_description_it_cannot_understand(self):
        """Fail closed, loudly, rather than scoring everyone identically."""
        import tempfile
        agent = ResumeScreeningAgent(mode="offline")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chef.md"
            path.write_text("# Chef de Partie\n\n## Required Qualifications\n"
                            "- Knife skills and a calm temperament\n")
            with self.assertRaises(ValueError):
                agent.load_job_description(path)

    def test_outputs_are_written_and_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_outputs(self.ranked, self.jd, Path(tmp))
            for path in paths.values():
                self.assertTrue(Path(path).exists())
            data = json.loads(Path(paths["json"]).read_text())
            self.assertEqual(data["candidate_count"], len(self.ranked))
            self.assertEqual(len(data["candidates"]), len(self.ranked))


if __name__ == "__main__":
    unittest.main()
