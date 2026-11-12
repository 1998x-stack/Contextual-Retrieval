import math

import pytest

from crlab.evaluation.metrics import aggregate, evaluate_query
from crlab.schemas import Query, SearchHit


def hit(identity, rank=1):
    return SearchHit(identity, 1.0, rank, "test")


def test_all_standard_scores_are_correct():
    q = Query("q", "why", ("a", "b"), ("A", "B"))
    metrics = evaluate_query(q, [hit("a"), hit("x", 2), hit("b", 3)],
                             {"a": "A", "b": "B", "x": "X"}, (1, 3))
    assert metrics["hit@1"] == 1.0
    assert metrics["recall@1"] == 0.5
    assert metrics["all_evidence@1"] == 0.0
    assert metrics["mrr@1"] == 1.0
    assert metrics["recall@3"] == 1.0
    assert metrics["all_evidence@3"] == 1.0
    expected_dcg = 1.0 + 1 / math.log2(4)
    ideal = 1.0 + 1 / math.log2(3)
    assert metrics["ndcg@3"] == pytest.approx(expected_dcg / ideal)


def test_legacy_content_match_is_distinct_from_id_recall():
    q = Query("q", "where", ("a", "b"), ("duplicate", "duplicate"))
    metrics = evaluate_query(q, [hit("a")], {"a": "duplicate", "b": "duplicate"}, (1,))
    assert metrics["recall@1"] == 0.5
    assert metrics["cookbook_pass@1"] == 1.0
    assert metrics["all_evidence@1"] == 0.0


def test_failing_query_is_included_in_macro_average():
    q = Query("q", "absent", ("a",), ("A",))
    metrics = evaluate_query(q, [], {"a": "A"}, (1,))
    assert metrics["recall@1"] == 0.0
    assert aggregate([{"recall@1": 1.0}, {"recall@1": metrics["recall@1"]}])["recall@1"] == 0.5


def test_invalid_cutoff_fails():
    q = Query("q", "x", ("a",), ("A",))
    with pytest.raises(ValueError, match="positive"):
        evaluate_query(q, [], {"a": "A"}, (0,))
