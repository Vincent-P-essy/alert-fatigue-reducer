"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
from .alert import AlertError, load
from .cluster import cluster as run_cluster
from .evaluate import evaluate, recommend, suppressed_alerts, sweep
from .score import Feedback, rank_scored, synthetic_corpus

SEVERITY_STYLE = {
    "critical": "bright_red", "high": "red", "medium": "yellow",
    "low": "cyan", "info": "dim",
}


def _alerts(args: argparse.Namespace):
    if getattr(args, "demo", False) or not getattr(args, "input", None):
        return synthetic_corpus(seed=args.seed, days=getattr(args, "days", 1))
    alerts = []
    for path in args.input:
        alerts.extend(load(path))
    return alerts


def _feedback(args: argparse.Namespace) -> Feedback | None:
    path = getattr(args, "feedback", None)
    return Feedback.load(path) if path and Path(path).exists() else None


def cmd_triage(args: argparse.Namespace, console: Console) -> int:
    alerts = _alerts(args)
    clusters = run_cluster(
        alerts, threshold=args.threshold, window=timedelta(hours=args.window)
    )
    scored = rank_scored(clusters, feedback=_feedback(args))
    result = evaluate(alerts, clusters, [s.cluster for s in scored], budget=args.budget)

    header = Text()
    header.append(f"{result.total_alerts} alerts  →  ", style="dim")
    header.append(f"{result.total_clusters} clusters", style="bold")
    header.append(f"   {result.volume_reduction:.1%} smaller queue\n", style="green")
    if result.true_positives:
        style = "green" if result.safe else "bright_red"
        header.append("true positives visible  ", style="dim")
        header.append(
            f"{result.tp_visible}/{result.true_positives}", style=f"bold {style}"
        )
        header.append(f"   in the top {result.budget}: ", style="dim")
        header.append(f"{result.tp_in_budget}/{result.true_positives}", style="bold")
    console.print(Panel(header, title="afr triage", border_style="blue", expand=False))

    if result.true_positives and not result.safe:
        console.print(
            f"[bold bright_red]{result.tp_suppressed} true positive(s) are no longer "
            "visible.[/] Lower --threshold before using this queue."
        )

    table = Table(
        title=f"Ranked queue (top {args.limit})", title_style="bold", header_style="dim"
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("score", justify="right")
    table.add_column("sev")
    table.add_column("×", justify="right", style="cyan")
    table.add_column("what it is", overflow="fold")
    table.add_column("why it is here", style="dim", overflow="fold")

    for index, item in enumerate(scored[: args.limit], start=1):
        rep = item.cluster.representative
        label = Text(item.cluster.summary())
        if item.cluster.contains_true_positive and args.show_labels:
            label.append("  ← real", style="bold bright_red")
        table.add_row(
            str(index),
            Text(f"{item.score:.0f}", style="bold"),
            Text(rep.severity, style=SEVERITY_STYLE.get(rep.severity, "white")),
            str(item.cluster.size),
            label,
            "; ".join(item.reasons[:3]),
        )
    console.print(table)

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "evaluation": result.to_dict(),
                    "clusters": [s.to_dict() for s in scored],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        console.print(f"[dim]wrote {args.json}[/]")

    return 0


def cmd_evaluate(args: argparse.Namespace, console: Console) -> int:
    """Sweep thresholds and show the trade-off rather than asserting one."""
    alerts = _alerts(args)
    labelled = sum(1 for a in alerts if a.label)
    if not labelled:
        console.print(
            "[bold red]this corpus has no labels.[/] Volume reduction without "
            "labelled data is unfalsifiable — deleting the queue reduces volume "
            "by 100%. Label a sample, or use --demo."
        )
        return 1

    results = sweep(alerts, window_hours=args.window, budget=args.budget)

    table = Table(
        title="What each threshold costs", title_style="bold", header_style="dim"
    )
    table.add_column("threshold", justify="right", style="bold")
    table.add_column("clusters", justify="right")
    table.add_column("reduction", justify="right")
    table.add_column("TP visible", justify="right")
    table.add_column(f"TP in top {args.budget}", justify="right")
    table.add_column("largest", justify="right", style="dim")
    table.add_column("verdict")
    for threshold, item in results:
        table.add_row(
            f"{threshold:.2f}",
            str(item.total_clusters),
            f"{item.volume_reduction:.1%}",
            f"{item.tp_visible}/{item.true_positives}",
            f"{item.tp_in_budget}/{item.true_positives}",
            str(item.largest_cluster),
            Text("safe", style="green") if item.safe
            else Text(f"hides {item.tp_suppressed}", style="bright_red"),
        )
    console.print(table)

    threshold, why = recommend(results)
    console.print(f"\n[bold]--threshold {threshold}[/] — {why}")

    worst = min(results, key=lambda pair: pair[1].tp_preservation)[1]
    if not worst.safe:
        console.print()
        clusters = run_cluster(
            alerts, threshold=max(t for t, _ in results),
            window=timedelta(hours=args.window),
        )
        hidden = suppressed_alerts(clusters)
        if hidden:
            table = Table(
                title="What the most aggressive threshold buried",
                title_style="bold", header_style="dim",
            )
            table.add_column("alert", style="bold")
            table.add_column("was merged into", overflow="fold")
            for group, alert in hidden[:8]:
                table.add_row(alert.message[:60], group.summary()[:70])
            console.print(table)
    return 0


def cmd_explain(args: argparse.Namespace, console: Console) -> int:
    alerts = _alerts(args)
    clusters = run_cluster(
        alerts, threshold=args.threshold, window=timedelta(hours=args.window)
    )
    scored = rank_scored(clusters, feedback=_feedback(args))
    match = next((s for s in scored if s.cluster.id == args.cluster), None)
    if match is None:
        console.print(f"[red]no cluster {args.cluster!r}[/] — ids look like C-0001")
        return 1

    console.print(
        Panel(
            Text.assemble(
                (match.cluster.summary() + "\n\n", "bold"),
                ("score ", "dim"), (f"{match.score:.0f}\n", "bold"),
                ("\n".join(f"  {r}" for r in match.reasons), ""),
            ),
            title=match.cluster.id,
            border_style="blue",
            expand=False,
        )
    )

    table = Table(title=f"{match.cluster.size} alerts", header_style="dim")
    table.add_column("time", style="dim")
    table.add_column("sev")
    table.add_column("source", style="cyan")
    table.add_column("target", style="cyan")
    table.add_column("message", overflow="fold")
    for alert in sorted(match.cluster.alerts, key=lambda a: a.timestamp)[:20]:
        table.add_row(
            alert.timestamp.strftime("%H:%M"),
            Text(alert.severity, style=SEVERITY_STYLE.get(alert.severity, "white")),
            alert.source, alert.target, alert.message[:70],
        )
    console.print(table)
    if match.cluster.size > 20:
        console.print(f"[dim]... and {match.cluster.size - 20} more[/]")
    return 0


def cmd_corpus(args: argparse.Namespace, console: Console) -> int:
    alerts = synthetic_corpus(seed=args.seed, days=args.days)
    path = Path(args.out)
    path.write_text(
        "\n".join(json.dumps(a.to_dict()) for a in alerts) + "\n", encoding="utf-8"
    )
    console.print(
        f"[green]wrote[/] {path} — {len(alerts)} alerts, "
        f"{sum(1 for a in alerts if a.is_true_positive)} labelled true positive"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="afr", description="Reduce alert volume without hiding the real ones."
    )
    parser.add_argument("--version", action="version", version=f"afr {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("input", nargs="*", help="JSON or NDJSON alert files")
        p.add_argument("--demo", action="store_true", help="use the bundled corpus")
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--days", type=int, default=1)
        p.add_argument("--window", type=int, default=6, help="clustering window, hours")
        p.add_argument("--budget", type=int, default=30, help="clusters an analyst works")
        p.add_argument("--feedback", help="disposition history JSON")

    p = sub.add_parser("triage", help="cluster, rank and show the queue")
    common(p)
    p.add_argument("--threshold", type=float, default=0.45)
    p.add_argument("--limit", type=int, default=15)
    p.add_argument("--show-labels", action="store_true", help="mark true positives")
    p.add_argument("--json", help="write the full result here")
    p.set_defaults(func=cmd_triage)

    p = sub.add_parser("evaluate", help="sweep thresholds against labelled data")
    common(p)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("explain", help="why a cluster ranks where it does")
    common(p)
    p.add_argument("cluster")
    p.add_argument("--threshold", type=float, default=0.45)
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("corpus", help="write the labelled demo corpus")
    p.add_argument("--out", default="corpus.ndjson")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--days", type=int, default=1)
    p.set_defaults(func=cmd_corpus)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()
    try:
        return int(args.func(args, console))
    except AlertError as exc:
        console.print(f"[bold red]input error:[/] {exc}")
        return 2
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]error:[/] {exc}")
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
