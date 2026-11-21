from absa.data import aspect_frequency, class_balance, domain_counts, review_lengths


def _record(review: str, aspect: str, sentiment: str, domain: str) -> dict[str, object]:
    return {"review": review, "aspect": aspect, "sentiment": sentiment, "domain": domain}


RECORDS = [
    _record("Battery is weak.", "battery", "negative", "laptop"),
    _record("Screen is great.", "screen", "positive", "laptop"),
    _record("Battery dies fast.", "battery", "negative", "laptop"),
    _record("Good food.", "food", "positive", "restaurant"),
]


def test_class_balance_counts_each_label() -> None:
    assert class_balance(RECORDS) == {"negative": 2, "positive": 2}


def test_aspect_frequency_orders_by_count_descending() -> None:
    assert aspect_frequency(RECORDS) == [("battery", 2), ("screen", 1), ("food", 1)]


def test_aspect_frequency_respects_top_n() -> None:
    assert aspect_frequency(RECORDS, top_n=1) == [("battery", 2)]


def test_domain_counts() -> None:
    assert domain_counts(RECORDS) == {"laptop": 3, "restaurant": 1}


def test_review_lengths_matches_input_order() -> None:
    lengths = review_lengths(RECORDS)
    assert lengths == [len(str(r["review"])) for r in RECORDS]


def test_empty_input_returns_empty_results() -> None:
    assert class_balance([]) == {}
    assert aspect_frequency([]) == []
    assert domain_counts([]) == {}
    assert review_lengths([]) == []
