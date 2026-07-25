"""Ranking the clusters, and the corpus used to check the ranking is honest.

Clustering shrinks the queue. Ranking decides what an analyst sees first, which
matters just as much: a 90% smaller queue is no help if the real incident is at
position 400.

The score is deliberately small and additive, so a rank can be explained to the
analyst who disagreed with it:

- **severity** — what the detection said
- **asset criticality** — what it is attached to
- **rarity** — a rule that fires 4,000 times a day carries less information per
  alert than one that fires twice a month
- **breadth** — one source hitting many targets is a sweep; many sources hitting
  one target is a distributed attempt. Both outrank a single pair.
- **disposition history** — a rule an analyst has closed as benign forty times
  should sink, and that is the only feedback signal here

Multiplicative scoring was tried and discarded: it produces a long tail of
near-zero scores where the ordering is noise, and it makes "why is this third?"
unanswerable.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .alert import Alert
from .cluster import Cluster

UTC = timezone.utc

SEVERITY_POINTS = {"critical": 40.0, "high": 25.0, "medium": 12.0, "low": 5.0, "info": 1.0}
CRITICALITY_POINTS = {"critical": 25.0, "high": 15.0, "medium": 6.0, "low": 2.0}


@dataclass
class Feedback:
    """What analysts decided about past alerts, by rule.

    The only learning in the system, and deliberately so: a rule closed as
    benign forty times running is the strongest signal available about what an
    analyst will do with the forty-first, and it needs no model to exploit.
    """

    dispositions: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, rule: str, disposition: str) -> None:
        bucket = self.dispositions.setdefault(rule, {"benign": 0, "true_positive": 0})
        if disposition in bucket:
            bucket[disposition] += 1

    def benign_rate(self, rule: str) -> float:
        bucket = self.dispositions.get(rule)
        if not bucket:
            return 0.0
        total = bucket["benign"] + bucket["true_positive"]
        return bucket["benign"] / total if total else 0.0

    def confidence(self, rule: str) -> int:
        bucket = self.dispositions.get(rule, {})
        return bucket.get("benign", 0) + bucket.get("true_positive", 0)

    @classmethod
    def load(cls, path: str | Path) -> Feedback:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(dispositions=data.get("dispositions", {}))

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(
            json.dumps({"dispositions": self.dispositions}, indent=2), encoding="utf-8"
        )
        return out


@dataclass(frozen=True)
class Scored:
    cluster: Cluster
    score: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 1),
            "reasons": list(self.reasons),
            **self.cluster.to_dict(),
        }


def score_cluster(
    cluster: Cluster,
    *,
    rule_frequency: dict[str, int] | None = None,
    total_alerts: int = 0,
    feedback: Feedback | None = None,
) -> Scored:
    rep = cluster.representative
    points = 0.0
    reasons: list[str] = []

    severity = SEVERITY_POINTS.get(rep.severity, 8.0)
    points += severity
    reasons.append(f"{rep.severity} severity (+{severity:.0f})")

    criticality = CRITICALITY_POINTS.get(rep.asset_criticality, 6.0)
    points += criticality
    if rep.asset_criticality in ("critical", "high"):
        reasons.append(f"{rep.asset_criticality} asset (+{criticality:.0f})")

    if rule_frequency and total_alerts:
        share = rule_frequency.get(rep.rule, 1) / total_alerts
        # A rule producing 40% of the queue carries little information per
        # alert; one producing 0.1% carries a lot.
        rarity = min(20.0, 4.0 / max(share, 0.002))
        points += rarity
        if rarity > 10:
            reasons.append(f"rare rule, {share:.1%} of the queue (+{rarity:.0f})")
        elif share > 0.15:
            reasons.append(f"noisy rule, {share:.0%} of the queue (+{rarity:.0f})")

    if len(cluster.targets) > 3:
        breadth = min(15.0, 3.0 * len(cluster.targets) ** 0.5)
        points += breadth
        reasons.append(f"{len(cluster.targets)} targets — looks like a sweep (+{breadth:.0f})")
    if len(cluster.sources) > 3:
        breadth = min(15.0, 3.0 * len(cluster.sources) ** 0.5)
        points += breadth
        reasons.append(f"{len(cluster.sources)} sources — distributed (+{breadth:.0f})")

    # Repetition against one critical target, from one or two sources, by one
    # user. Individually these are low-severity failures; together they are
    # what credential stuffing looks like, and scoring only the representative
    # buries the cluster at the bottom of the queue. This was found by the
    # budget-recall metric, not by reading the code.
    if (
        cluster.size >= 5
        and len(cluster.targets) <= 1
        and len(cluster.sources) <= 2
        and rep.asset_criticality in ("critical", "high")
    ):
        persistence = min(20.0, 4.0 * cluster.size**0.5)
        points += persistence
        reasons.append(
            f"{cluster.size} attempts against one {rep.asset_criticality} asset "
            f"— consistent with brute force (+{persistence:.0f})"
        )

    if feedback:
        confidence = feedback.confidence(rep.rule)
        if confidence >= 5:
            benign = feedback.benign_rate(rep.rule)
            penalty = -25.0 * benign
            points += penalty
            if benign > 0.7:
                reasons.append(
                    f"closed benign {benign:.0%} of {confidence} times ({penalty:.0f})"
                )

    return Scored(cluster=cluster, score=max(0.0, points), reasons=tuple(reasons))


def rank(
    clusters: list[Cluster],
    *,
    feedback: Feedback | None = None,
) -> list[Cluster]:
    """Clusters in the order an analyst should work them."""
    return [s.cluster for s in rank_scored(clusters, feedback=feedback)]


def rank_scored(
    clusters: list[Cluster],
    *,
    feedback: Feedback | None = None,
) -> list[Scored]:
    total = sum(c.size for c in clusters)
    frequency: dict[str, int] = {}
    for cluster in clusters:
        for alert in cluster.alerts:
            frequency[alert.rule] = frequency.get(alert.rule, 0) + 1

    scored = [
        score_cluster(
            cluster, rule_frequency=frequency, total_alerts=total, feedback=feedback
        )
        for cluster in clusters
    ]
    scored.sort(key=lambda s: (-s.score, -s.cluster.size, s.cluster.id))
    return scored


# -- corpus -----------------------------------------------------------------


def synthetic_corpus(seed: int = 7, days: int = 1) -> list[Alert]:
    """A labelled corpus with the shape of a real SOC queue.

    Deterministic, so a published reduction figure can be re-derived. The
    proportions are the point: a handful of rules produce most of the volume,
    almost all of it benign, and the few true positives are scattered inside
    that flood rather than conveniently at the top.
    """
    rng = random.Random(seed)
    start = datetime(2026, 7, 20, 0, 0, tzinfo=UTC)
    alerts: list[Alert] = []
    counter = 0

    def add(**kwargs: Any) -> None:
        nonlocal counter
        counter += 1
        alerts.append(Alert(id=f"a-{counter:05d}", **kwargs))

    # The vulnerability scanner nobody remembered to allowlist: one source,
    # many targets, hundreds of alerts, all benign. The single biggest source
    # of L1 fatigue in most SOCs.
    scanner = "10.20.4.19"
    for index in range(320 * days):
        add(
            rule="Port scan detected",
            timestamp=start + timedelta(minutes=index * 2 % 1440, days=index // 720),
            severity="medium", source=scanner,
            target=f"10.30.{index % 4}.{index % 250}",
            category="recon",
            message=f"Multiple connection attempts from {scanner} to port {22 + index % 30}",
            asset_criticality="low", label="benign",
        )

    # A misconfigured backup agent failing authentication all night.
    for index in range(140 * days):
        add(
            rule="Failed authentication",
            timestamp=start + timedelta(minutes=index * 6 % 1440, days=index // 240),
            severity="low", source="10.20.9.42", target="fileserver-01",
            user="svc_backup", category="authentication",
            message="Failed login for svc_backup from 10.20.9.42 (bad password)",
            asset_criticality="medium", label="benign",
        )

    # Ordinary background noise across many rules and hosts.
    rules = [
        ("Outbound connection to new domain", "medium", "network"),
        ("PowerShell execution", "medium", "execution"),
        ("Privilege escalation attempt", "high", "privilege"),
        ("Suspicious DNS query", "low", "network"),
        ("Unusual process parent", "medium", "execution"),
    ]
    for index in range(180 * days):
        rule, severity, category = rules[index % len(rules)]
        host = f"wks-{rng.randint(1, 400):03d}"
        add(
            rule=rule,
            timestamp=start + timedelta(minutes=rng.randint(0, 1440 * days)),
            severity=severity, source=f"10.40.{rng.randint(0, 8)}.{rng.randint(1, 250)}",
            target=host, user=f"u.{rng.choice('abcdefghijk')}{rng.randint(100, 999)}",
            category=category,
            message=f"{rule} on {host}",
            asset_criticality=rng.choice(["low", "medium", "medium", "high"]),
            label="benign",
        )

    # The real incident: credential stuffing that succeeds, then lateral
    # movement. Buried inside the flood, on a critical asset, and deliberately
    # *not* the highest-volume rule.
    attacker = "203.0.113.77"
    for index in range(6):
        add(
            rule="Failed authentication",
            timestamp=start + timedelta(hours=3, minutes=index * 3),
            severity="low", source=attacker, target="vpn-gw-01",
            user="m.dubois", category="authentication",
            message=f"Failed login for m.dubois from {attacker}",
            asset_criticality="critical", label="true_positive",
        )
    add(
        rule="Successful authentication from new geography",
        timestamp=start + timedelta(hours=3, minutes=20),
        severity="high", source=attacker, target="vpn-gw-01", user="m.dubois",
        category="authentication",
        message=f"m.dubois signed in from {attacker} (first seen from RO)",
        asset_criticality="critical", label="true_positive",
    )
    add(
        rule="Privilege escalation attempt",
        timestamp=start + timedelta(hours=3, minutes=41),
        severity="high", source="10.30.1.14", target="pay-core-02", user="m.dubois",
        category="privilege",
        message="m.dubois attempted to add themselves to Domain Admins",
        asset_criticality="critical", label="true_positive",
    )
    add(
        rule="Data staged for exfiltration",
        timestamp=start + timedelta(hours=4, minutes=12),
        severity="critical", source="10.30.1.14", target="pay-core-02", user="m.dubois",
        category="exfiltration",
        message="4.2 GB archived to C:\\Windows\\Temp\\bk.7z by m.dubois",
        asset_criticality="critical", label="true_positive",
    )

    alerts.sort(key=lambda a: a.timestamp)
    return alerts
