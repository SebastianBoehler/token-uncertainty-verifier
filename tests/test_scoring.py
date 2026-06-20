from token_uncertainty.scoring import analyze_scored_tokens, claim_cues_for_text, split_sentence_spans
from token_uncertainty.types import TokenScore


def test_claim_cues_find_legal_and_factual_markers():
    cues = claim_cues_for_text("In 1897, Brown v. Board cited Section 14 of the Act.")

    assert "date" in cues
    assert "case-name" in cues
    assert "section" in cues
    assert "legal-term" in cues


def test_low_probability_claim_token_gets_high_risk():
    result = analyze_scored_tokens(
        [
            TokenScore(text="In ", probability=0.9, entropy=0.1, margin=0.3),
            TokenScore(text="1897", probability=0.05, entropy=0.8, margin=-0.2),
            TokenScore(text=", the court ruled.", probability=0.5, entropy=0.4, margin=0.0),
        ]
    )

    risky = result.tokens[1]
    assert risky.uncertainty > 0.7
    assert risky.factual_risk > 0.7
    assert result.sentences[0].factual_risk > 0.5


def test_empty_analysis_has_no_sentences():
    result = analyze_scored_tokens([])

    assert result.text == ""
    assert result.tokens == []
    assert result.sentences == []


def test_legal_abbreviations_do_not_split_sentences():
    spans = split_sentence_spans("Roe v. Wade, 410 U.S. 113, was decided in 1973. Next.")

    assert len(spans) == 2
    assert spans[0][2] == "Roe v. Wade, 410 U.S. 113, was decided in 1973."
