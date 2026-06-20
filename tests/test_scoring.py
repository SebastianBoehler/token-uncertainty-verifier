from token_uncertainty.scoring import analyze_scored_tokens, claim_cues_for_text, split_sentence_spans
from token_uncertainty.types import TokenScore


def test_claim_cues_find_factual_markers():
    cues = claim_cues_for_text("In 1969, Apollo 11 landed near Site 2 according to report 14.")

    assert "date" in cues
    assert "named-entity" in cues
    assert "reference" in cues
    assert "claim-term" in cues


def test_low_probability_claim_token_gets_high_risk():
    result = analyze_scored_tokens(
        [
            TokenScore(text="In ", probability=0.9, entropy=0.1, margin=0.3),
            TokenScore(text="1897", probability=0.05, entropy=0.8, margin=-0.2),
            TokenScore(text=", the observatory opened.", probability=0.5, entropy=0.4, margin=0.0),
        ]
    )

    risky = result.tokens[1]
    assert risky.uncertainty > 0.7
    assert risky.factual_risk > 0.7
    assert result.sentences[0].factual_risk > 0.5


def test_sentence_cues_do_not_color_every_token():
    result = analyze_scored_tokens(
        [
            TokenScore(text="The summary said ", probability=0.8, entropy=0.1, margin=0.2),
            TokenScore(text="Apollo", probability=0.2, entropy=0.7, margin=-0.1),
            TokenScore(text=" in ", probability=0.8, entropy=0.1, margin=0.2),
            TokenScore(text="1972", probability=0.2, entropy=0.7, margin=-0.1),
            TokenScore(text=".", probability=0.8, entropy=0.1, margin=0.2),
        ]
    )

    assert result.tokens[0].claim_cues == ()
    assert result.tokens[2].claim_cues == ()
    assert result.tokens[2].factual_risk < 0.2
    assert result.tokens[3].claim_cues == ("date", "number")
    assert result.tokens[3].factual_risk > 0.7


def test_repeated_subject_with_conflicting_dates_flags_values():
    result = analyze_scored_tokens(
        [
            TokenScore(text="Apollo 11 landed in ", probability=0.8, entropy=0.1, margin=0.2),
            TokenScore(text="1969", probability=0.8, entropy=0.1, margin=0.2),
            TokenScore(text=". Apollo 11 landed in ", probability=0.8, entropy=0.1, margin=0.2),
            TokenScore(text="1972", probability=0.8, entropy=0.1, margin=0.2),
            TokenScore(text=".", probability=0.8, entropy=0.1, margin=0.2),
        ]
    )

    conflicted = [token for token in result.tokens if "conflict" in token.claim_cues]

    assert len(conflicted) == 2
    assert all("date" in token.claim_cues for token in conflicted)
    assert all(token.factual_risk > 0.9 for token in conflicted)


def test_distinct_numbers_in_one_sentence_are_not_conflicts():
    result = analyze_scored_tokens(
        [
            TokenScore(
                text="Project Atlas released build ATLAS-42 in report 14.",
                probability=0.8,
                entropy=0.1,
                margin=0.2,
            )
        ]
    )

    assert all("conflict" not in token.claim_cues for token in result.tokens)


def test_empty_analysis_has_no_sentences():
    result = analyze_scored_tokens([])

    assert result.text == ""
    assert result.tokens == []
    assert result.sentences == []


def test_common_abbreviations_do_not_split_sentences():
    spans = split_sentence_spans("The U.S. mission launched in 1969. Next.")

    assert len(spans) == 2
    assert spans[0][2] == "The U.S. mission launched in 1969."
