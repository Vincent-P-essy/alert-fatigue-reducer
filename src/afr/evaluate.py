"""The measurement that decides whether any of this was a good idea.

Every deduplication tool reduces volume. Volume reduction on its own is not a
result — deleting the queue reduces volume by 100%. The number that matters is
**how many true positives stopped being visible**, and it is the number these
tools almost never report.

An alert is *visible* after clustering if it is the representative of its
cluster, or if its cluster surfaces high enough in the ranked queue that an
analyst reaches it. A true positive buried inside a cluster of 300 benign
alerts, whose representative is one of the benign ones, has been suppressed
even though nothing was deleted.

So this module reports three things together, and refuses to report the first
without the other two:

- **volume reduction** — how much smaller the queue got
- **true positives preserved** — how many remained visible
- **budget recall** — of the true positives, how many appear in the top *N*
  clusters an analyst will actually work through in a shift
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .alert import Alert
from .cluster import Cluster


@dataclass(frozen=True)
class Evaluation:
    total_alerts: int
    total_clusters: int
    true_positives: int
    tp_visible: int
    tp_suppressed: int
    largest_cluster: int
    singletons: int
    budget: int
    tp_in_budget: int

    @property
    def volume_reduction(self) -> float:
        if not self.total_alerts:
            return 0.0
        return 1 - (self.total_clusters / self.total_alerts)

    @property
    def tp_preservation(self) -> float:
        """The number that has to be 1.0 before anyone deploys this."""
        if not self.true_positives:
            return 1.0
        return self.tp_visible / self.true_positives

    @property
    def budget_recall(self) -> float:
        """True positives reachable in one shift's worth of triage."""
        if not self.true_positives:
            return 1.0
        return self.tp_in_budget / self.true_positives

    @property
    def safe(self) -> bool:
        return self.tp_suppressed == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_alerts": self.total_alerts,
            "total_clusters": self.total_clusters,
            "volume_reduction": round(self.volume_reduction, 4),
            "true_positives": self.true_positives,
            "true_positives_visible": self.tp_visible,
            "true_positives_suppressed": self.tp_suppressed,
            "tp_preservation": round(self.tp_preservation, 4),
            "triage_budget": self.budget,
            "true_positives_in_budget": self.tp_in_budget,
            "budget_recall": round(self.budget_recall, 4),
            "largest_cluster": self.largest_cluster,
            "singletons": self.singletons,
            "safe": self.safe,
        }


def evaluate(
    alerts: list[Alert],
    clusters: list[Cluster],
    ranked: list[Cluster] | None = None,
    *,
    budget: int = 30,
) -> Evaluation:
    """Measure the reduction and what it cost.

    ``ranked`` is the queue in the order an analyst would work it; when it is
    supplied, budget recall answers "would they have got to the real ones".
    """
    true_positives = [a for a in alerts if a.is_true_positive]

    visible = 0
    for cluster in clusters:
        cluster_tps = sum(1 for a in cluster.alerts if a.is_true_positive)
        if not cluster_tps:
            continue
        if cluster.representative.is_true_positive:
            # The analyst opens this cluster because its representative is the
            # real alert, and finds everything in it. Grouping six alerts of
            # one incident into one row is the desired behaviour, not
            # suppression - counting it as loss would penalise the tool for
            # working.
            visible += cluster_tps
        # A true positive inside a cluster whose representative is benign is
        # *not* visible: the analyst reads the summary, sees a benign event,
        # and moves on. Counting it would make the headline number look good
        # in exactly the case where the tool has done harm.

    tp_in_budget = 0
    if ranked is not None:
        for cluster in ranked[:budget]:
            if cluster.contains_true_positive:
                tp_in_budget += sum(1 for a in cluster.alerts if a.is_true_positive)

    return Evaluation(
        total_alerts=len(alerts),
        total_clusters=len(clusters),
        true_positives=len(true_positives),
        tp_visible=visible,
        tp_suppressed=len(true_positives) - visible if true_positives else 0,
        largest_cluster=max((c.size for c in clusters), default=0),
        singletons=sum(1 for c in clusters if c.size == 1),
        budget=budget,
        tp_in_budget=tp_in_budget,
    )


def suppressed_alerts(clusters: list[Cluster]) -> list[tuple[Cluster, Alert]]:
    """Every true positive that stopped being visible, and where it went.

    A tuning tool needs this: "reduction went from 71% to 84% and buried two
    real incidents" is actionable, "reduction went to 84%" is not.
    """
    out: list[tuple[Cluster, Alert]] = []
    for cluster in clusters:
        if cluster.representative.is_true_positive:
            continue
        for alert in cluster.alerts:
            if alert.is_true_positive:
                out.append((cluster, alert))
    return out


def sweep(
    alerts: list[Alert],
    thresholds: tuple[float, ...] = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85),
    *,
    window_hours: int = 6,
    budget: int = 30,
) -> list[tuple[float, Evaluation]]:
    """Evaluate several thresholds, so the trade-off is visible rather than assumed.

    The right threshold is not a constant: it depends on the detection estate,
    the alert mix and how much triage capacity exists. What a tool can do is
    show the curve and refuse to pick a point on it that suppresses real
    incidents.
    """
    from datetime import timedelta

    from .cluster import cluster as run_cluster
    from .score import rank

    out = []
    for threshold in thresholds:
        clusters = run_cluster(
            alerts, threshold=threshold, window=timedelta(hours=window_hours)
        )
        out.append(
            (threshold, evaluate(alerts, clusters, rank(clusters), budget=budget))
        )
    return out


def recommend(results: list[tuple[float, Evaluation]]) -> tuple[float, str]:
    """The most aggressive threshold that suppresses nothing real.

    Deliberately not "the one with the best reduction". A tool that optimises
    volume and reports the suppression separately is one whose default
    configuration hides incidents.
    """
    safe = [(t, e) for t, e in results if e.safe]
    if not safe:
        best = max(results, key=lambda pair: pair[1].tp_preservation)
        return best[0], (
            "every threshold tested suppressed at least one true positive. "
            "Do not deploy this until the feature weights are re-tuned for "
            "your detection estate."
        )
    chosen = max(safe, key=lambda pair: pair[1].volume_reduction)
    unsafe = [t for t, e in results if not e.safe]
    note = (
        f" Thresholds {', '.join(f'{t:g}' for t in unsafe)} suppressed real alerts."
        if unsafe
        else " No threshold tested suppressed anything real, so this is the "
        "best reduction available rather than a safety compromise."
    )
    return chosen[0], (
        f"{chosen[1].volume_reduction:.0%} volume reduction with every true "
        f"positive still visible.{note}"
    )
