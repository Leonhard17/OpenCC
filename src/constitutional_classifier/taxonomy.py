"""Harm taxonomy for the constitutional classifier.

Kept standalone (mirrors TACTIC's ``tactic/taxonomy.py``) so OpenCC has no dependency on
the training package. A loaded model's own taxonomy — read from its ``weight_frame.json`` —
always takes precedence over these defaults; this list is only the fallback ordering.
"""

from __future__ import annotations

TAXONOMY: list[str] = [
    "Benign",  # index 0
    "CBRN",  # Chemical, Biological, Radiological, Nuclear
    "Cyber",
    "Harm to Minors",
    "Harmful Manipulation",
    "Hate Speech",
    "Illegal Activities",
    "Information Integrity",
    "Physical Harm",
    "Privacy",
    "Self-Harm",
    "Sexual Content",
]

# Harm categories only (the multilabel objective has one sigmoid head per harm category;
# "Benign" means nothing fired).
HARM_CATEGORIES: list[str] = [c for c in TAXONOMY if c != "Benign"]
