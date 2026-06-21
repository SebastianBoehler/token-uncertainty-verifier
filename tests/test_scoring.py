from token_uncertainty.scoring import analyze_scored_tokens, split_sentence_spans
from token_uncertainty.types import TokenScore


def test_low_probability_token_gets_high_uncertainty():
    result = analyze_scored_tokens(
        [
            TokenScore(text="In ", probability=0.9, entropy=0.1, margin=0.3),
            TokenScore(text="1897", probability=0.05, entropy=0.8, margin=-0.2),
            TokenScore(text=", the observatory opened.", probability=0.5, entropy=0.4, margin=0.0),
        ]
    )

    uncertain = result.tokens[1]
    assert uncertain.uncertainty > 0.7
    assert result.sentences[0].uncertainty_score > 0.5


def test_high_confidence_token_stays_low_uncertainty():
    result = analyze_scored_tokens(
        [
            TokenScore(text="Stable", probability=0.95, entropy=0.05, margin=0.3),
            TokenScore(text=" text.", probability=0.9, entropy=0.1, margin=0.2),
        ]
    )

    assert all(token.uncertainty < 0.25 for token in result.tokens)
    assert result.sentences[0].uncertainty_score < 0.25


def test_scoring_does_not_attach_rule_based_labels():
    result = analyze_scored_tokens(
        [
            TokenScore(text="Apollo 11 landed in ", probability=0.8, entropy=0.1, margin=0.2),
            TokenScore(text="1969", probability=0.8, entropy=0.1, margin=0.2),
            TokenScore(text=".", probability=0.8, entropy=0.1, margin=0.2),
        ]
    )

    removed_attr = "claim" + "_cues"
    assert not hasattr(result.tokens[0], removed_attr)
    assert not hasattr(result.sentences[0], removed_attr)


def test_sentence_scores_summarize_token_uncertainty():
    result = analyze_scored_tokens(
        [
            TokenScore(text="One.", probability=0.95, entropy=0.05, margin=0.3),
            TokenScore(text=" Two", probability=0.05, entropy=0.8, margin=-0.2),
            TokenScore(text=".", probability=0.9, entropy=0.1, margin=0.2),
        ]
    )

    assert len(result.sentences) == 2
    assert result.sentences[1].max_token_uncertainty > result.sentences[0].max_token_uncertainty
    assert result.sentences[1].uncertainty_score > result.sentences[0].uncertainty_score


def test_empty_analysis_has_no_sentences():
    result = analyze_scored_tokens([])

    assert result.text == ""
    assert result.tokens == []
    assert result.sentences == []


def test_common_abbreviations_do_not_split_sentences():
    spans = split_sentence_spans("The U.S. mission launched in 1969. Next.")

    assert len(spans) == 2
    assert spans[0][2] == "The U.S. mission launched in 1969."
