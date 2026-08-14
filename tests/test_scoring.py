"""Tests for the scoring model. These pin down the policy decisions."""
import unittest

from src.config import WEIGHTS
from src.schema import JobDescription, ResumeProfile
from src.scoring import (education_fit, experience_fit, score_candidate,
                         skill_coverage)


def make_jd(**kw):
    defaults = dict(title="ML Engineer",
                    required_skills=["python", "pytorch", "nlp", "docker"],
                    preferred_skills=["aws", "kubernetes"],
                    min_years_experience=3.0, min_degree_level=2,
                    raw_text="python pytorch nlp docker aws kubernetes")
    defaults.update(kw)
    return JobDescription(**defaults)


def make_profile(**kw):
    defaults = dict(candidate_id="C001", name="Test", years_experience=3.0,
                    degree_level=2, skills=["python", "pytorch", "nlp", "docker"],
                    raw_text="python pytorch nlp docker")
    defaults.update(kw)
    return ResumeProfile(**defaults)


class TestExperienceCurve(unittest.TestCase):
    def test_meeting_the_bar_scores_075_not_1(self):
        """Meeting the minimum is a pass, not a perfect score."""
        self.assertAlmostEqual(experience_fit(3.0, 3.0), 0.75)

    def test_saturates_at_one(self):
        self.assertAlmostEqual(experience_fit(7.0, 3.0), 1.0)
        self.assertAlmostEqual(experience_fit(25.0, 3.0), 1.0)

    def test_extra_experience_has_diminishing_returns(self):
        """The gain from 3->5 years must exceed the gain from 9->11."""
        early = experience_fit(5.0, 3.0) - experience_fit(3.0, 3.0)
        late = experience_fit(11.0, 3.0) - experience_fit(9.0, 3.0)
        self.assertGreater(early, late)

    def test_below_the_bar_is_linear_not_zero(self):
        """A near-miss candidate must stay rankable, not be zeroed out."""
        self.assertGreater(experience_fit(2.5, 3.0), 0.0)
        self.assertLess(experience_fit(2.5, 3.0), 0.75)

    def test_no_requirement_means_full_marks(self):
        self.assertEqual(experience_fit(0.0, 0.0), 1.0)


class TestEducationFit(unittest.TestCase):
    def test_exceeding_gives_no_bonus(self):
        """A PhD must not out-score a Bachelor's when a Bachelor's was asked for."""
        self.assertEqual(education_fit(4, 2), education_fit(2, 2))

    def test_each_level_short_costs_04(self):
        self.assertAlmostEqual(education_fit(1, 2), 0.6)
        self.assertAlmostEqual(education_fit(0, 2), 0.2)


class TestSkillCoverage(unittest.TestCase):
    def test_all_required_no_preferred(self):
        jd, profile = make_jd(), make_profile()
        score, matched, missing, _ = skill_coverage(jd, profile)
        self.assertEqual(missing, [])
        self.assertEqual(len(matched), 4)
        # 4 required / (4 + 0.35*2 preferred) = 0.8511
        self.assertAlmostEqual(score, 4 / (4 + 0.35 * 2), places=3)

    def test_preferred_cannot_replace_a_missing_required(self):
        jd = make_jd()
        has_all_required = make_profile()
        trades_required_for_preferred = make_profile(
            skills=["python", "pytorch", "nlp", "aws", "kubernetes"])
        self.assertGreater(skill_coverage(jd, has_all_required)[0],
                           skill_coverage(jd, trades_required_for_preferred)[0])


class TestFailsClosed(unittest.TestCase):
    def test_uncovered_jd_scores_zero_not_one(self):
        """A JD the taxonomy does not cover must not give everyone full marks.

        This previously returned 1.0, so an empty resume scored 0.88 against a
        chef vacancy and was labelled a strong match.
        """
        jd = make_jd(required_skills=[], preferred_skills=[])
        empty = make_profile(skills=[], years_experience=0.0, degree_level=0)
        self.assertEqual(skill_coverage(jd, empty)[0], 0.0)


class TestScoreComposition(unittest.TestCase):
    def test_weights_renormalise_without_the_llm(self):
        """Offline mode must rescale to 1.0, not silently cap everyone."""
        breakdown = score_candidate(make_jd(), make_profile(), semantic=0.5)
        self.assertNotIn("llm_judgment", breakdown.components)
        self.assertAlmostEqual(sum(breakdown.weights_used.values()), 1.0, places=6)
        self.assertFalse(breakdown.llm_used)

    def test_weights_sum_to_one_with_the_llm(self):
        breakdown = score_candidate(make_jd(), make_profile(), semantic=0.5,
                                    llm_judgment=0.8)
        self.assertIn("llm_judgment", breakdown.components)
        self.assertAlmostEqual(sum(breakdown.weights_used.values()), 1.0, places=6)
        self.assertTrue(breakdown.llm_used)

    def test_perfect_candidate_approaches_one(self):
        jd = make_jd()
        perfect = make_profile(years_experience=10.0, degree_level=3,
                               skills=jd.required_skills + jd.preferred_skills)
        breakdown = score_candidate(jd, perfect, semantic=1.0)
        self.assertGreater(breakdown.final_score, 0.99)

    def test_empty_candidate_scores_near_zero(self):
        breakdown = score_candidate(make_jd(),
                                    make_profile(skills=[], years_experience=0.0,
                                                 degree_level=0),
                                    semantic=0.0)
        self.assertLess(breakdown.final_score, 0.05)

    def test_score_is_bounded(self):
        for years in (0, 3, 40):
            for semantic in (0.0, 0.5, 1.0):
                s = score_candidate(make_jd(),
                                    make_profile(years_experience=years),
                                    semantic=semantic).final_score
                self.assertTrue(0.0 <= s <= 1.0)

    def test_declared_weights_match_config(self):
        """Guards against config drift silently changing every score."""
        b = score_candidate(make_jd(), make_profile(), semantic=0.5, llm_judgment=0.5)
        for key, weight in b.weights_used.items():
            self.assertAlmostEqual(weight, WEIGHTS[key], places=6)


if __name__ == "__main__":
    unittest.main()
