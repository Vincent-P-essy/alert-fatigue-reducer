"""MinHash correctness, clustering behaviour, and the measurement that matters."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from afr.alert import (
    Alert,
    AlertError,
    from_record,
    load,
    shingles,
    weighted_jaccard,
)
from afr.cluster import (
    NUM_PERMUTATIONS,
    candidate_pairs,
    cluster,
    minhash,
    signature_similarity,
)
from afr.evaluate import evaluate, recommend, suppressed_alerts, sweep
from afr.score import Feedback, rank, rank_scored, score_cluster, synthetic_corpus

UTC = timezone.utc
BASE = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)


def make(rule="Port scan detected", *, minute=0, source="10.0.0.1", target="10.0.1.5",
         severity="medium", user="", message="", label="", criticality="medium",
         alert_id=None):
    return Alert(
        id=alert_id or f"a-{minute}-{source}-{target}",
        rule=rule, timestamp=BASE + timedelta(minutes=minute), severity=severity,
        source=source, target=target, user=user,
        message=message or f"{rule} from {source}", label=label,
        asset_criticality=criticality,
    )


class TestMinHash:
    def test_identical_sets_have_identical_signatures(self):
        tokens = {"a", "b", "c"}
        assert minhash(tokens) == minhash(set(tokens))

    def test_signature_is_stable_across_processes(self):
        # Python's built-in hash() is randomised per process; a clustering
        # result that changes between runs is not one anyone can debug.
        assert minhash({"rule0:port scan"})[0] == minhash({"rule0:port scan"})[0]

    def test_estimates_jaccard(self):
        left = {f"t{i}" for i in range(100)}
        right = {f"t{i}" for i in range(50, 150)}  # true Jaccard = 50/150 = 0.333
        estimate = signature_similarity(minhash(left), minhash(right))
        assert 0.20 < estimate < 0.47  # within MinHash error at k=128

    def test_disjoint_sets_estimate_near_zero(self):
        left = {f"a{i}" for i in range(50)}
        right = {f"b{i}" for i in range(50)}
        assert signature_similarity(minhash(left), minhash(right)) < 0.1

    def test_empty_set(self):
        assert len(minhash(set())) == NUM_PERMUTATIONS

    def test_signature_length(self):
        assert len(minhash({"x"})) == NUM_PERMUTATIONS


class TestSimilarity:
    def test_shingles_catch_reordered_wording(self):
        a = shingles("failed login for alice from 10.0.0.1")
        b = shingles("login failed for alice from 10.0.0.1")
        assert a & b

    def test_identical_alerts_score_one(self):
        assert weighted_jaccard(make(), make()) == pytest.approx(1.0)

    def test_different_rules_score_low(self):
        # The rule carries the most weight: two alerts from different rules are
        # rarely the same event however much else they share.
        assert weighted_jaccard(make(rule="Port scan"), make(rule="Data exfiltration")) < 0.7

    def test_same_source_different_target_still_matches(self):
        # One attacker sweeping a subnet must not become 254 clusters.
        score = weighted_jaccard(
            make(target="10.0.1.5", message="scan"), make(target="10.0.9.99", message="scan")
        )
        assert score > 0.6

    def test_different_source_same_target_matches_less(self):
        score = weighted_jaccard(
            make(source="10.0.0.1", message="x"), make(source="10.0.0.2", message="x")
        )
        assert score < 0.85


class TestClustering:
    def test_a_scanner_collapses_into_one_cluster(self):
        alerts = [
            make(minute=i, target=f"10.0.1.{i}", message="Port scan detected")
            for i in range(40)
        ]
        clusters = cluster(alerts, threshold=0.45)
        assert len(clusters) == 1
        assert clusters[0].size == 40

    def test_distinct_events_stay_distinct(self):
        alerts = [
            make(rule="Port scan detected", message="scan"),
            make(rule="Data exfiltration", message="4 GB uploaded", severity="critical"),
            make(rule="Malware detected", message="trojan quarantined", severity="high"),
        ]
        assert len(cluster(alerts, threshold=0.45)) == 3

    def test_time_window_separates_repeats(self):
        # The same rule on the same host today and three weeks ago are not one
        # event; similarity alone has no way to know that.
        alerts = [make(minute=0), make(minute=60 * 24 * 21, alert_id="later")]
        assert len(cluster(alerts, window=timedelta(hours=6), threshold=0.45)) == 2

    def test_representative_is_the_worst_then_the_earliest(self):
        alerts = [
            make(minute=5, severity="low", alert_id="low"),
            make(minute=10, severity="critical", alert_id="crit"),
            make(minute=1, severity="medium", alert_id="med"),
        ]
        group = cluster(alerts, threshold=0.3)[0]
        assert group.representative.id == "crit"

    def test_empty_input(self):
        assert cluster([]) == []

    def test_single_alert(self):
        clusters = cluster([make()])
        assert len(clusters) == 1 and clusters[0].size == 1

    def test_summary_replaces_hundreds_of_rows(self):
        alerts = [make(minute=i, target=f"10.0.1.{i}") for i in range(30)]
        summary = cluster(alerts, threshold=0.45)[0].summary()
        assert "×30" in summary
        assert "10.0.0.1" in summary
        assert "targets" in summary

    def test_lsh_proposes_far_fewer_than_every_pair(self):
        alerts = synthetic_corpus()
        pairs = candidate_pairs(sorted(alerts, key=lambda a: a.timestamp))
        every_pair = len(alerts) * (len(alerts) - 1) // 2
        assert len(pairs) < every_pair * 0.5

    def test_clustering_is_deterministic(self):
        alerts = synthetic_corpus()
        first = [c.size for c in cluster(alerts, threshold=0.45)]
        second = [c.size for c in cluster(alerts, threshold=0.45)]
        assert first == second


class TestCorpus:
    def test_has_the_shape_of_a_real_queue(self):
        alerts = synthetic_corpus()
        assert len(alerts) > 500
        true_positives = [a for a in alerts if a.is_true_positive]
        # A handful of real incidents inside a flood - the ratio is the point.
        assert 5 <= len(true_positives) <= 15
        assert len(true_positives) / len(alerts) < 0.03

    def test_is_deterministic(self):
        assert [a.id for a in synthetic_corpus()] == [a.id for a in synthetic_corpus()]

    def test_different_seeds_differ(self):
        assert synthetic_corpus(seed=1) != synthetic_corpus(seed=2)

    def test_true_positives_are_not_conveniently_at_the_top(self):
        # If they were the loudest alerts the exercise would be trivial.
        alerts = synthetic_corpus()
        first_tp = next(i for i, a in enumerate(alerts) if a.is_true_positive)
        assert first_tp > 20


@pytest.fixture(scope="module")
def corpus():
    return synthetic_corpus()


class TestEvaluation:
    def test_reduction_is_substantial(self, corpus):
        clusters = cluster(corpus, threshold=0.45)
        result = evaluate(corpus, clusters, rank(clusters))
        assert result.volume_reduction > 0.7

    def test_no_true_positive_is_suppressed_at_the_recommended_threshold(self, corpus):
        threshold, _ = recommend(sweep(corpus))
        clusters = cluster(corpus, threshold=threshold)
        result = evaluate(corpus, clusters, rank(clusters))
        assert result.tp_suppressed == 0
        assert result.safe

    def test_a_correctly_grouped_incident_is_not_counted_as_suppression(self):
        # Six alerts of one brute-force attempt collapsing into one row is the
        # desired behaviour. Counting five of them as "hidden" would penalise
        # the tool for working.
        alerts = [
            make(minute=i, rule="Failed authentication", user="victim",
                 target="vpn-01", source="203.0.113.7", severity="low",
                 label="true_positive", alert_id=f"tp-{i}")
            for i in range(6)
        ]
        clusters = cluster(alerts, threshold=0.45)
        assert len(clusters) == 1
        result = evaluate(alerts, clusters)
        assert result.tp_visible == 6
        assert result.tp_suppressed == 0

    def test_a_true_positive_buried_under_benign_alerts_is_suppression(self):
        alerts = [
            make(minute=i, severity="medium", message="Port scan detected",
                 target=f"10.0.1.{i}", alert_id=f"benign-{i}")
            for i in range(30)
        ]
        alerts.append(
            make(minute=5, severity="low", message="Port scan detected",
                 target="10.0.1.99", label="true_positive", alert_id="real")
        )
        clusters = cluster(alerts, threshold=0.45)
        result = evaluate(alerts, clusters)
        assert result.tp_suppressed == 1
        assert not result.safe
        assert suppressed_alerts(clusters)

    def test_volume_reduction_alone_is_never_reported(self, corpus):
        # Deleting the queue reduces volume by 100%.
        clusters = cluster(corpus, threshold=0.45)
        payload = evaluate(corpus, clusters, rank(clusters)).to_dict()
        assert "true_positives_suppressed" in payload
        assert "budget_recall" in payload

    def test_sweep_shows_the_trade_off(self, corpus):
        results = sweep(corpus)
        assert len(results) >= 4
        reductions = [e.volume_reduction for _, e in results]
        # More aggressive thresholds cannot reduce less.
        assert reductions[0] >= reductions[-1]

    def test_recommendation_prefers_safety_over_reduction(self):
        from afr.evaluate import Evaluation

        aggressive = Evaluation(100, 5, 4, 2, 2, 50, 1, 30, 2)   # 95% but hides 2
        safe = Evaluation(100, 40, 4, 4, 0, 10, 20, 30, 4)       # 60% and hides none
        threshold, why = recommend([(0.3, aggressive), (0.7, safe)])
        assert threshold == 0.7
        assert "still visible" in why

    def test_recommendation_refuses_when_nothing_is_safe(self):
        from afr.evaluate import Evaluation

        bad = Evaluation(100, 5, 4, 1, 3, 50, 1, 30, 1)
        _, why = recommend([(0.3, bad)])
        assert "Do not deploy" in why

    def test_serialises(self, corpus):
        clusters = cluster(corpus, threshold=0.45)
        json.dumps(evaluate(corpus, clusters, rank(clusters)).to_dict())
        json.dumps([s.to_dict() for s in rank_scored(clusters)])


class TestRanking:
    def test_real_incidents_reach_the_analyst(self):
        corpus = synthetic_corpus()
        clusters = cluster(corpus, threshold=0.45)
        result = evaluate(corpus, clusters, rank(clusters), budget=30)
        # The whole point: a smaller queue is no help if the real thing is at
        # position 400.
        assert result.budget_recall == 1.0

    def test_repeated_low_severity_attempts_on_a_critical_asset_rank_up(self):
        # Credential stuffing is many low-severity failures. Scoring only the
        # representative buries it; the metric found this, not a code review.
        alerts = [
            make(minute=i, rule="Failed authentication", severity="low",
                 target="vpn-01", source="203.0.113.7", user="victim",
                 criticality="critical", alert_id=f"bf-{i}")
            for i in range(8)
        ]
        group = cluster(alerts, threshold=0.45)[0]
        scored = score_cluster(group)
        assert any("brute force" in r for r in scored.reasons)

    def test_every_score_carries_its_reasons(self):
        corpus = synthetic_corpus()
        for item in rank_scored(cluster(corpus, threshold=0.45)):
            assert item.reasons

    def test_feedback_sinks_a_rule_analysts_keep_closing(self):
        feedback = Feedback()
        for _ in range(20):
            feedback.record("Port scan detected", "benign")

        alerts = [make(minute=i, target=f"10.0.1.{i}") for i in range(10)]
        group = cluster(alerts, threshold=0.45)[0]
        without = score_cluster(group).score
        with_history = score_cluster(group, feedback=feedback).score
        assert with_history < without

    def test_feedback_needs_evidence_before_it_acts(self):
        # Two closures are not a pattern.
        feedback = Feedback()
        feedback.record("Port scan detected", "benign")
        feedback.record("Port scan detected", "benign")
        group = cluster([make()], threshold=0.45)[0]
        assert score_cluster(group, feedback=feedback).score == score_cluster(group).score

    def test_feedback_round_trips(self, tmp_path):
        feedback = Feedback()
        feedback.record("r", "benign")
        restored = Feedback.load(feedback.save(tmp_path / "fb.json"))
        assert restored.benign_rate("r") == 1.0


class TestIngest:
    def test_ndjson(self, tmp_path):
        path = tmp_path / "a.ndjson"
        path.write_text(
            "\n".join(
                json.dumps({"rule": "r", "timestamp": "2026-07-20T12:00:00Z", "id": str(i)})
                for i in range(3)
            )
        )
        assert len(load(path)) == 3

    def test_json_array(self, tmp_path):
        path = tmp_path / "a.json"
        path.write_text(json.dumps([{"rule": "r", "timestamp": "2026-07-20T12:00:00Z"}]))
        assert len(load(path)) == 1

    def test_wrapped_object(self, tmp_path):
        path = tmp_path / "a.json"
        path.write_text(json.dumps({"alerts": [{"rule": "r", "timestamp": "2026-07-20T12:00:00Z"}]}))
        assert len(load(path)) == 1

    def test_rule_is_required(self):
        with pytest.raises(AlertError, match="'rule' is required"):
            from_record({"timestamp": "2026-07-20T12:00:00Z"})

    def test_field_aliases(self):
        alert = from_record({
            "signature": "r", "time": "2026-07-20T12:00:00Z",
            "src_ip": "1.2.3.4", "dest_ip": "5.6.7.8", "username": "u",
        })
        assert alert.rule == "r" and alert.source == "1.2.3.4"
        assert alert.target == "5.6.7.8" and alert.user == "u"

    def test_naive_timestamps_get_utc(self):
        assert from_record({"rule": "r", "timestamp": "2026-07-20T12:00:00"}).timestamp.tzinfo

    def test_bad_timestamp(self):
        with pytest.raises(AlertError, match="cannot parse timestamp"):
            from_record({"rule": "r", "timestamp": "yesterday"})

    def test_missing_file(self, tmp_path):
        with pytest.raises(AlertError, match="not found"):
            load(tmp_path / "nope.json")
