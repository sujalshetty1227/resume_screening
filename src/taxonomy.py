"""Deterministic skill detection via a canonical skill -> alias taxonomy.

This is the component that makes scores defensible: "candidate is missing
Docker" is a checkable claim about the resume text, not an LLM opinion.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

from .config import TAXONOMY_PATH
from .text_utils import normalize


@lru_cache(maxsize=1)
def load_taxonomy(path: str = str(TAXONOMY_PATH)) -> Dict[str, List[str]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _alias_pattern(alias: str) -> re.Pattern:
    """Boundary-aware regex for one alias.

    Plain \\b fails on 'c++' (the '+' is already a non-word character, so \\b
    after it never matches). Explicit lookarounds over the technical character
    class handle c++, c#, node.js and ci/cd correctly.

    Short aliases get two extra restrictions, because one- and two-letter skill
    names ('R', 'Go', 'ML') are the ones that generate false positives:

      * Strict boundaries additionally exclude '-', '.' and '&', so
        "Go-to-market", "R-squared", "R. K. Sharma" and "R&D" do not register
        as programming languages. Longer aliases keep the permissive boundary
        so that "Python-based" still matches `python`.
      * CASE SENSITIVITY. 'R' and 'Go' are skills only when capitalised;
        lowercase "I go to work" and "the r value" are ordinary English.
        Boundaries alone cannot separate those two, because the token really
        is identical -- only the capitalisation differs. This is why
        find_skills searches the ORIGINAL text for short aliases and the
        normalised text for everything else.
      * Pluralisation is disabled, since it made 'ml' match "MLS" and 'r'
        match "Rs.".
    """
    escaped = re.escape(alias).replace(r"\ ", r"[\s\-_]+")

    if len(alias) <= 2:
        variants = "|".join({alias.upper(), alias.capitalize()})
        return re.compile(rf"(?<![A-Za-z0-9+#.&\-])(?:{variants})"
                          rf"(?![A-Za-z0-9+#&.\-])")

    # Optional plural: the JD says "vector databases", the taxonomy says
    # "vector database".
    return re.compile(rf"(?<![a-z0-9+#.&]){escaped}(?:e?s)?(?![a-z0-9+#&])",
                      re.IGNORECASE)


def _is_short(alias: str) -> bool:
    return len(alias) <= 2


@lru_cache(maxsize=1)
def _compiled_taxonomy() -> Dict[str, List[Tuple[re.Pattern, bool]]]:
    """skill -> [(pattern, needs_original_case), ...]"""
    return {skill: [(_alias_pattern(a), _is_short(a)) for a in aliases]
            for skill, aliases in load_taxonomy().items()}


def find_skills(text: str) -> List[str]:
    """Canonical skills evidenced anywhere in the text, sorted for stability.

    Two haystacks: short aliases are matched case-sensitively against the
    original text, everything else case-insensitively against the normalised
    text. See _alias_pattern for why.
    """
    lowered = normalize(text)
    found = {skill for skill, patterns in _compiled_taxonomy().items()
             if any(p.search(text if cased else lowered) for p, cased in patterns)}
    return sorted(found)


def canonicalize(skill_names: List[str]) -> List[str]:
    """Map free-text skill strings (e.g. from the LLM) onto canonical names.

    Anything unrecognised is kept as-is rather than dropped: the LLM may
    legitimately spot a skill the taxonomy does not cover yet.
    """
    taxonomy = load_taxonomy()
    alias_to_canon = {a.lower(): canon
                      for canon, aliases in taxonomy.items() for a in aliases}
    alias_to_canon.update({c.lower(): c for c in taxonomy})

    out: List[str] = []
    for name in skill_names:
        key = normalize(str(name))
        out.append(alias_to_canon.get(key, key))
    return sorted(set(s for s in out if s))
