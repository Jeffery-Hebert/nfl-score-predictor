"""
The standing rule -- broken logic is ALWAYS fixed, even if the model gets less
accurate -- must stay written down everywhere someone decides what to change.

Operator rule, 2026-10-02. It was written into the README (for the public:
junior dbt analytics engineers, high-school readers), the development skill and
its findings log (for whoever develops next), the promotion gate's own
docstring, and the ledger page. This pins every copy, so an edit cannot quietly
drop one and leave the others disagreeing.

Run: pytest tests/test_correctness_rule_documented.py -v
"""

from pathlib import Path

import pytest

SKILL = Path(".claude/skills/nfl-prediction-architect")

# file -> phrases that must appear in it (case-insensitive)
PLACES = {
    "README.md": [
        "correct before accurate",
        "even if the fix makes the model\nless accurate",
        "broken logic is fixed, not voted on",
        "never used to decide whether to fix broken logic",
    ],
    str(SKILL / "SKILL.md"): [
        "correctness before accuracy",
        "if the logic is broken it must be fixed",
        "fixes are never gated on accuracy",
    ],
    str(SKILL / "references/project-context-and-findings.md"): [
        "standing rule: correctness before accuracy",
        "if the logic is broken it must be fixed",
    ],
    "src/validate/calibration.py": ["correct before accurate"],
    "src/predict/report_template.html": [
        "broken logic always gets fixed, even if the model gets less accurate"
    ],
}


@pytest.mark.parametrize("path", sorted(PLACES))
def test_the_rule_is_stated(path):
    text = Path(path).read_text().lower()
    for phrase in PLACES[path]:
        assert phrase.lower() in text, f"{path} no longer states: {phrase!r}"


def test_the_readme_states_it_before_any_accuracy_number_is_explained():
    """Section 1 -- read before anything else, by the readers it is for."""
    text = Path("README.md").read_text()
    rule = text.index("The one rule above accuracy: correct before accurate")
    assert rule < text.index("## 2. Vocabulary you'll need")


def test_the_skill_puts_it_above_the_objective_priority():
    text = (SKILL / "SKILL.md").read_text()
    assert text.index("## Correctness Before Accuracy") < text.index(
        "## Objective Priority"
    )
