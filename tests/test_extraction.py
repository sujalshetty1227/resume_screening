"""Tests for parsing and rule-based extraction."""
import unittest

from src.extraction import (detect_degree_level, estimate_years_experience,
                            extract_jd_rule, extract_phone, extract_resume_rule,
                            guess_name, strip_education_lines)
from src.taxonomy import canonicalize, find_skills


class TestYearsExperience(unittest.TestCase):
    def test_union_span_not_sum_of_overlapping_roles(self):
        """Concurrent roles must not be double-counted.

        Three roles totalling 9 role-years span only 6 calendar years.
        Summing would report 9 and inflate every freelancer in the pool.
        """
        text = ("Engineer, Acme 2018 - 2021\n"
                "Consultant, Beta 2019 - 2022\n"
                "Advisor, Gamma 2021 - 2024")
        self.assertEqual(estimate_years_experience(text), 6.0)

    def test_present_resolves_to_current_year(self):
        self.assertEqual(estimate_years_experience("Engineer 2021 - Present"), 5.0)

    def test_falls_back_to_self_reported_claim(self):
        self.assertEqual(estimate_years_experience("6 years of experience"), 6.0)

    def test_dates_take_priority_over_an_inflated_claim(self):
        text = "15 years of experience\nEngineer, Acme 2022 - 2025"
        self.assertEqual(estimate_years_experience(text), 3.0)

    def test_absurd_values_rejected(self):
        self.assertEqual(estimate_years_experience("120 years of experience"), 0.0)

    def test_no_signal_returns_zero(self):
        self.assertEqual(estimate_years_experience("I like building things."), 0.0)


class TestDegree(unittest.TestCase):
    def test_highest_degree_wins(self):
        self.assertEqual(detect_degree_level("B.Tech 2019\nPh.D. 2025"), 4)

    def test_abbreviations(self):
        self.assertEqual(detect_degree_level("M.Tech in CS"), 3)
        self.assertEqual(detect_degree_level("B.E. in Electronics"), 2)
        self.assertEqual(detect_degree_level("Diploma in Engineering"), 1)

    def test_no_degree(self):
        self.assertEqual(detect_degree_level("Self-taught developer"), 0)


class TestDegreeFalsePositives(unittest.TestCase):
    """Regression tests for the bug where `b\\.?e\\.?` matched the word "be"."""

    def test_common_english_words_are_not_degrees(self):
        for text in ["I would be delighted to be considered for this role.",
                     "Proficient in MS Office and MS Excel.",
                     "Would be a good fit for the BS of day-to-day delivery."]:
            self.assertEqual(detect_degree_level(text), 0, text)

    def test_undotted_abbreviations_recovered_by_context(self):
        """"MS in Data Science" is a degree; "MS Office" is not."""
        self.assertEqual(detect_degree_level("MS in Data Science"), 3)
        self.assertEqual(detect_degree_level("BS in Computer Science"), 2)
        self.assertEqual(detect_degree_level("MA in Linguistics"), 3)
        self.assertEqual(detect_degree_level("MCA"), 3)
        self.assertEqual(detect_degree_level("BEng Mechanical"), 2)
        self.assertEqual(detect_degree_level("latency of 50 ms in production"), 0)
        self.assertEqual(detect_degree_level("we will be in charge"), 0)

    def test_real_abbreviations_still_detected(self):
        self.assertEqual(detect_degree_level("B.E. in Information Science"), 2)
        self.assertEqual(detect_degree_level("B.A. in Linguistics"), 2)
        self.assertEqual(detect_degree_level("M.Sc. in Statistics"), 3)
        self.assertEqual(detect_degree_level("M.S. in Electrical Engineering"), 3)
        self.assertEqual(detect_degree_level("B.Tech in CS"), 2)


class TestPhone(unittest.TestCase):
    def test_international_groupings(self):
        """The 5+5 Indian mobile grouping broke the previous 3+4 regex."""
        for raw in ["+91 98450 11234", "+92 300 1234567", "+49 151 2233 4455",
                    "+234 802 555 1199", "+65 8123 4567"]:
            self.assertEqual(extract_phone(f"Contact: {raw}"), raw)

    def test_non_phone_numbers_rejected(self):
        for text in ["CGPA 9.1/10", "Engineer 2019 - 2023", "300+ GitHub stars",
                     "GPA 3.85/4.00 (2016-2020)", "AWS account 123456789012"]:
            self.assertIsNone(extract_phone(text), text)


class TestSkillMatching(unittest.TestCase):
    def test_aliases_resolve_to_canonical_names(self):
        self.assertIn("pytorch", find_skills("Experienced with Torch"))
        self.assertIn("transformers", find_skills("Used HuggingFace daily"))
        self.assertIn("gcp", find_skills("Deployed on Google Cloud"))

    def test_plural_surface_forms_match(self):
        self.assertIn("vector database", find_skills("We use vector databases"))

    def test_punctuation_heavy_tokens(self):
        self.assertIn("c++", find_skills("Strong C++ background"))
        self.assertIn("javascript", find_skills("Node.js services"))

    def test_no_substring_false_positives(self):
        """'R' and 'Go' are real taxonomy aliases, so this test has teeth.

        (It previously did not: neither bare alias existed, so it passed
        against any regex at all.)
        """
        from src.taxonomy import load_taxonomy
        self.assertIn("r", load_taxonomy()["r"])
        self.assertIn("go", load_taxonomy()["go"])

        found = find_skills("Reporting to the CTO. Googled a lot. I am going home.")
        self.assertNotIn("r", found)
        self.assertNotIn("go", found)

    def test_single_letter_skills_survive_adjacent_punctuation(self):
        """Short aliases need strict boundaries AND case sensitivity."""
        for text in ["R&D lead; paid Rs. 5000 per hour", "Improved R-squared to 0.92",
                     "the R-CNN paper", "R. K. Sharma", "Go-to-market strategy",
                     "I go to work", "e-commerce go-live"]:
            found = find_skills(text)
            self.assertNotIn("r", found, text)
            self.assertNotIn("go", found, text)
        self.assertIn("r", find_skills("Strong R and Go experience"))
        self.assertIn("go", find_skills("Built with Go and Rust"))

    def test_long_aliases_keep_permissive_boundaries(self):
        """The strict boundary must not break "Python-based"."""
        self.assertIn("python", find_skills("Python-based microservices"))
        self.assertIn("scikit-learn", find_skills("Used scikit-learn heavily"))

    def test_removed_ambiguous_aliases(self):
        self.assertNotIn("computer vision", find_skills("Curriculum Vitae (CV) attached"))
        self.assertNotIn("aws", find_skills("Managed lambdas in Python"))

    def test_plural_rule_does_not_apply_to_short_aliases(self):
        """'ml' + plural matched 'MLS'; the plural rule is off below 3 chars."""
        self.assertNotIn("machine learning", find_skills("MLS property listings"))

    def test_unknown_skills_are_kept_not_dropped(self):
        self.assertIn("quantum annealing", canonicalize(["Quantum Annealing"]))


class TestJobDescription(unittest.TestCase):
    def test_degree_lines_do_not_become_required_skills(self):
        """'Bachelor's in ... Statistics' must not make statistics a skill."""
        line = "Bachelor's degree in Computer Science, Statistics or related field"
        self.assertNotIn("statistics", find_skills(strip_education_lines(line)))
        self.assertIn("statistics", find_skills(line))   # unstripped, for contrast

    def test_required_and_preferred_are_separated(self):
        jd = extract_jd_rule(
            "# Role\n\n## Required Qualifications\n- Python and PyTorch\n"
            "- 4+ years of experience\n\n## Preferred Qualifications\n- Kubernetes\n")
        self.assertIn("python", jd.required_skills)
        self.assertIn("kubernetes", jd.preferred_skills)
        self.assertNotIn("kubernetes", jd.required_skills)
        self.assertEqual(jd.min_years_experience, 4.0)


class TestResumeProfile(unittest.TestCase):
    def test_contact_details(self):
        p = extract_resume_rule("Jane Doe\njane.doe@example.com\n+91 98450 11234\n"
                                "Python and Docker", "C001", "jane.txt")
        self.assertEqual(p.email, "jane.doe@example.com")
        self.assertEqual(p.name, "Jane Doe")
        self.assertIn("python", p.skills)

    def test_name_guess_skips_headers_with_digits_or_emails(self):
        self.assertEqual(guess_name("221B Baker Street\nx@y.com\nSherlock Holmes"),
                         "Sherlock Holmes")

    def test_empty_resume_does_not_crash(self):
        p = extract_resume_rule("", "C001", "empty.txt")
        self.assertEqual(p.skills, [])
        self.assertEqual(p.years_experience, 0.0)
        self.assertEqual(p.name, "Unknown")


if __name__ == "__main__":
    unittest.main()
