"""Clustering: MinHash + LSH banding, then an exact check.

Comparing every alert against every other is quadratic, which is fine for a
thousand alerts and hopeless for a day's worth from a real SIEM. So candidate
pairs come from **locality-sensitive hashing**: alerts are reduced to MinHash
signatures, the signature is cut into bands, and two alerts become candidates
only if some band matches exactly. Two alerts with Jaccard similarity *s* land
in the same bucket with probability `1 - (1 - s^r)^b`, which is an S-curve — a
sharp, tunable threshold rather than a fuzzy one.

Candidates are then scored **exactly** with the weighted Jaccard from
:mod:`afr.alert`, because the decision to merge two alerts should rest on the
real number and not on a hash collision.

Two constraints that come from the domain rather than the algorithm:

**A time window.** The same rule firing on the same host today and three weeks
ago are not one event. Similarity has no notion of time; the window supplies it.

**Single linkage, deliberately.** A scanning host produces a chain of alerts
that drift gradually, and transitive merging is what collapses the chain.
Complete linkage would leave the flood in place, which is the flood the tool
exists to remove.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta

from .alert import Alert, weighted_jaccard

#: 128 permutations. Signature error falls as 1/sqrt(k); 128 puts it around 9%,
#: which is well inside the margin the exact re-check corrects for.
NUM_PERMUTATIONS = 128
#: 32 bands of 4 rows. The S-curve inflects around (1/b)^(1/r) = 0.42, so pairs
#: above ~0.5 similarity are reliably proposed and pairs below ~0.3 rarely are.
NUM_BANDS = 32
ROWS_PER_BAND = NUM_PERMUTATIONS // NUM_BANDS

#: A Mersenne prime, so the affine permutations below are a universal family.
_PRIME = (1 << 61) - 1
_MAX = _PRIME


def _base_hash(token: str) -> int:
    """One deterministic 61-bit hash per token.

    blake2b rather than Python's ``hash``, which is randomised per process - a
    clustering result that changes between runs is not one anyone can debug.
    """
    return int.from_bytes(
        hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big"
    ) % _PRIME


def _coefficients(permutations: int) -> tuple[tuple[int, int], ...]:
    """Fixed (a, b) pairs for h_i(x) = (a_i·x + b_i) mod p.

    The obvious implementation hashes every token once per permutation, which
    is 128 blake2b calls per token and dominates the runtime - about 2.5M
    hashes for a day of alerts. Hashing once and permuting with a universal
    family gives the same collision probabilities for a fraction of the work.
    """
    rng = random.Random(20260720)  # fixed: signatures must be stable across runs
    return tuple(
        (rng.randrange(1, _PRIME), rng.randrange(0, _PRIME)) for _ in range(permutations)
    )


_COEFFICIENTS = _coefficients(NUM_PERMUTATIONS)


def minhash(tokens: set[str], permutations: int = NUM_PERMUTATIONS) -> tuple[int, ...]:
    """The MinHash signature of a token set."""
    if not tokens:
        return tuple([_MAX] * permutations)
    coefficients = (
        _COEFFICIENTS if permutations == NUM_PERMUTATIONS else _coefficients(permutations)
    )
    bases = [_base_hash(token) for token in tokens]
    return tuple(
        min((a * base + b) % _PRIME for base in bases) for a, b in coefficients
    )


def signature_similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    """Estimated Jaccard: the fraction of matching signature positions."""
    if not left or not right:
        return 0.0
    return sum(1 for a, b in zip(left, right, strict=True) if a == b) / len(left)


@dataclass
class Cluster:
    """A group of alerts judged to be the same event."""

    id: str
    alerts: list[Alert] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.alerts)

    @property
    def representative(self) -> Alert:
        """The one an analyst should look at.

        The highest-severity alert, then the earliest — an analyst wants the
        worst thing in the group and the moment it started, not whichever
        happened to arrive first.
        """
        order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        return min(
            self.alerts,
            key=lambda a: (order.get(a.severity, 5), a.timestamp),
        )

    @property
    def first_seen(self):
        return min(a.timestamp for a in self.alerts)

    @property
    def last_seen(self):
        return max(a.timestamp for a in self.alerts)

    @property
    def span(self) -> timedelta:
        return self.last_seen - self.first_seen

    @property
    def sources(self) -> set[str]:
        return {a.source for a in self.alerts if a.source}

    @property
    def targets(self) -> set[str]:
        return {a.target for a in self.alerts if a.target}

    @property
    def rules(self) -> set[str]:
        return {a.rule for a in self.alerts}

    @property
    def contains_true_positive(self) -> bool:
        return any(a.is_true_positive for a in self.alerts)

    def summary(self) -> str:
        """What an analyst reads instead of 400 rows."""
        rep = self.representative
        if self.size == 1:
            return rep.message or rep.rule

        parts = [f"{rep.rule} ×{self.size}"]
        if len(self.sources) == 1:
            parts.append(f"from {next(iter(self.sources))}")
        elif len(self.sources) > 1:
            parts.append(f"from {len(self.sources)} sources")
        if len(self.targets) > 1:
            parts.append(f"against {len(self.targets)} targets")
        elif self.targets:
            parts.append(f"against {next(iter(self.targets))}")
        minutes = self.span.total_seconds() / 60
        parts.append(f"over {minutes:.0f} min" if minutes >= 1 else "within a minute")
        return " ".join(parts)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "size": self.size,
            "summary": self.summary(),
            "representative": self.representative.to_dict(),
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "rules": sorted(self.rules),
            "sources": sorted(self.sources),
            "targets": sorted(self.targets)[:20],
            "alert_ids": [a.id for a in self.alerts],
        }


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def candidate_pairs(alerts: list[Alert]) -> set[tuple[int, int]]:
    """Index-pairs worth checking exactly, via LSH banding."""
    signatures = [minhash(a.shingle_set()) for a in alerts]
    buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)

    for index, signature in enumerate(signatures):
        for band in range(NUM_BANDS):
            start = band * ROWS_PER_BAND
            key = (band, *signature[start : start + ROWS_PER_BAND])
            buckets[key].append(index)

    pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        # A bucket holding most of the corpus means the signature carries no
        # information for this data - pairing it up would recreate the
        # quadratic comparison LSH exists to avoid.
        if len(members) > 200:
            continue
        for i, left in enumerate(members):
            for right in members[i + 1 :]:
                pairs.add((left, right) if left < right else (right, left))
    return pairs


def cluster(
    alerts: list[Alert],
    *,
    threshold: float = 0.55,
    window: timedelta = timedelta(hours=6),
) -> list[Cluster]:
    """Group alerts that are the same event.

    ``threshold`` is the exact weighted-Jaccard score two alerts must reach.
    ``window`` bounds how far apart in time two alerts can be and still merge.
    """
    if not alerts:
        return []

    ordered = sorted(alerts, key=lambda a: a.timestamp)
    union = _UnionFind(len(ordered))

    for left, right in candidate_pairs(ordered):
        if abs(ordered[right].timestamp - ordered[left].timestamp) > window:
            continue
        if weighted_jaccard(ordered[left], ordered[right]) >= threshold:
            union.union(left, right)

    groups: dict[int, list[Alert]] = defaultdict(list)
    for index, alert in enumerate(ordered):
        groups[union.find(index)].append(alert)

    clusters = [
        Cluster(id=f"C-{number:04d}", alerts=members)
        for number, members in enumerate(
            sorted(groups.values(), key=lambda g: (-len(g), g[0].timestamp)), start=1
        )
    ]
    return clusters


def reduction(alerts: list[Alert], clusters: list[Cluster]) -> float:
    """Fraction of the queue removed. Meaningless on its own — see afr.evaluate."""
    if not alerts:
        return 0.0
    return 1 - (len(clusters) / len(alerts))
