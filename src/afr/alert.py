"""The alert model, and the features clustering actually works on.

Feature selection is the whole game here, and it is a security question rather
than a machine-learning one.

Include the source IP and 400 alerts from one scanning host collapse into one
cluster — which is right. Include the *timestamp* at full precision and nothing
ever matches anything. Include the destination IP and a single attacker
sweeping a /24 becomes 254 separate clusters, which is exactly the flood the
tool exists to remove.

So features are weighted, and the weights encode judgements a security analyst
would recognise: the rule that fired matters most, the actor next, the specific
target least. Getting this wrong does not produce a crash — it produces a tool
that quietly hides real attacks, which is why the weights are named constants
with reasons rather than magic numbers inside a function.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc


class AlertError(ValueError):
    """Raised when alert input cannot be read."""


#: Weights used when comparing two alerts. Higher means "this being the same
#: is stronger evidence that these are the same event".
FEATURE_WEIGHTS = {
    # The rule is the strongest signal: two alerts from different rules are
    # rarely the same event however much else they share.
    "rule": 3.0,
    # Who is doing it. A scanning host produces hundreds of near-identical
    # alerts and collapsing them is the point.
    "source": 2.0,
    "user": 2.0,
    # What is being hit. Weighted low deliberately: one attacker sweeping a
    # subnet must not become 254 separate clusters.
    "target": 0.5,
    # Free text, shingled. Catches variants of the same rule wording.
    "text": 1.0,
    "category": 1.0,
}


@dataclass
class Alert:
    """One alert, as a detection tool emitted it."""

    id: str
    rule: str
    timestamp: datetime
    severity: str = "medium"
    source: str = ""
    target: str = ""
    user: str = ""
    category: str = ""
    message: str = ""
    asset_criticality: str = "medium"
    raw: dict[str, Any] = field(default_factory=dict)
    #: Ground truth, when the corpus has it. Never used by clustering or
    #: scoring - only by evaluation, which is the point of keeping it separate.
    label: str = ""

    @property
    def is_true_positive(self) -> bool:
        return self.label == "true_positive"

    def features(self) -> dict[str, set[str]]:
        """The weighted feature sets used for similarity."""
        return {
            "rule": {self.rule.lower()} if self.rule else set(),
            "source": {self.source.lower()} if self.source else set(),
            "user": {self.user.lower()} if self.user else set(),
            "target": {self.target.lower()} if self.target else set(),
            "category": {self.category.lower()} if self.category else set(),
            "text": shingles(self.message),
        }

    def shingle_set(self) -> set[str]:
        """One flat, weighted token set, for MinHash.

        Weight is applied by *repeating* a token under distinct prefixes. It
        looks crude and it is exactly right for MinHash: the probability of two
        signatures colliding is the Jaccard similarity of the sets, so
        repeating a token raises its influence on that similarity in proportion.
        """
        out: set[str] = set()
        for name, values in self.features().items():
            weight = FEATURE_WEIGHTS.get(name, 1.0)
            repeats = max(1, int(weight * 2))
            for value in values:
                for index in range(repeats):
                    out.add(f"{name}{index}:{value}")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "rule": self.rule,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity,
            "source": self.source,
            "target": self.target,
            "user": self.user,
            "category": self.category,
            "message": self.message,
            "asset_criticality": self.asset_criticality,
            "label": self.label,
        }


_WORD = re.compile(r"[a-z0-9_.\-]+")


def shingles(text: str, size: int = 3) -> set[str]:
    """Word 3-grams. Catches "failed login for X" against "login failed for X".

    Character shingles would match more aggressively and also match unrelated
    messages that happen to share substrings, which is the wrong error to make
    in a tool whose failure mode is hiding real alerts.
    """
    if not text:
        return set()
    words = _WORD.findall(text.lower())
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def weighted_jaccard(left: Alert, right: Alert) -> float:
    """Similarity in [0, 1], computed feature by feature.

    Used for the exact score once MinHash has proposed a candidate pair.
    Cheaper approximations are fine for candidate generation; the decision to
    merge two alerts should be made on the real number.
    """
    left_features, right_features = left.features(), right.features()
    total = 0.0
    matched = 0.0

    for name, weight in FEATURE_WEIGHTS.items():
        a, b = left_features.get(name, set()), right_features.get(name, set())
        if not a and not b:
            continue
        union = a | b
        if not union:
            continue
        total += weight
        matched += weight * (len(a & b) / len(union))

    return matched / total if total else 0.0


def parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise AlertError(f"cannot parse timestamp {value!r}") from None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def from_record(record: dict[str, Any], index: int = 0) -> Alert:
    if not isinstance(record, dict):
        raise AlertError(f"alert {index}: expected an object")
    rule = record.get("rule") or record.get("rule_name") or record.get("signature")
    if not rule:
        raise AlertError(f"alert {index}: 'rule' is required")

    return Alert(
        id=str(record.get("id") or f"alert-{index}"),
        rule=str(rule),
        timestamp=parse_timestamp(record.get("timestamp") or record.get("time") or "1970-01-01"),
        severity=str(record.get("severity", "medium")).lower(),
        source=str(record.get("source") or record.get("src_ip") or ""),
        target=str(record.get("target") or record.get("dest_ip") or record.get("host") or ""),
        user=str(record.get("user") or record.get("username") or ""),
        category=str(record.get("category", "")),
        message=str(record.get("message") or record.get("description") or ""),
        asset_criticality=str(record.get("asset_criticality", "medium")).lower(),
        label=str(record.get("label", "")),
        raw=record,
    )


def load(path: str | Path) -> list[Alert]:
    """Read JSON, NDJSON or a wrapped object."""
    source = Path(path)
    if not source.exists():
        raise AlertError(f"alert file not found: {source}")

    text = source.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text.startswith("["):
        records = json.loads(text)
    elif text.startswith("{") and "\n" not in text.strip():
        payload = json.loads(text)
        records = payload.get("alerts", [payload])
    else:
        records = []
        for number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AlertError(f"{source.name}:{number}: {exc}") from None

    return [from_record(record, index) for index, record in enumerate(records)]
