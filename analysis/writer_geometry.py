"""Compare remaining projections and relative weight changes across writers."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, median

CONDITIONS = ("s1_then_s2", "s2_then_s1", "joint")
LABELS = ("S1 then S2", "S2 then S1", "Joint")
COLORS = ("#2563a6", "#bb601f", "#21806b")
WRITERS = ("embed",) + tuple(f"{kind}_{i}" for i in range(26) for kind in ("o_proj", "down_proj"))


def load_writers(root):
    panel = {}
    for condition in CONDITIONS:
        path = root / "results" / "smoke" / f"{condition}.writers.json"
        raw = json.loads(path.read_text())
        if set(raw) != set(WRITERS):
            raise ValueError(f"Unexpected writer inventory: {path}")
        rows = {}
        for name in WRITERS:
            row = raw[name]
            ratios = row["relative_to_original"]
            if len(ratios) != 2:
                raise ValueError(f"Expected two directional ratios: {condition}/{name}")
            values = tuple(100 * v for v in (*ratios, row["relative_writer_change"]))
            if any(not math.isfinite(v) or v < 0 for v in values):
                raise ValueError(f"Invalid measurement: {condition}/{name}")
            rows[name] = values
        panel[condition] = rows
    return panel


def summary(panel):
    lines = [
        "# Writer-level geometry",
        "",
        "Reanalysis of the archived Gemma 2 2B Instruct pilot. No new model runs.",
        "",
        "Each row below aggregates 53 effective residual-writing matrices. Directional ratios use the original norm for the same writer and direction. Weight changes use each original writer's Frobenius norm.",
        "",
        "| Condition | Median S1 remaining | Max S1 remaining | Median S2 remaining | Max S2 remaining | Mean weight change |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, label in zip(CONDITIONS, LABELS):
        s1, s2, change = zip(*(panel[condition][name] for name in WRITERS))
        lines.append(f"| {label} | {median(s1):.3f}% | {max(s1):.3f}% | {median(s2):.3f}% | {max(s2):.3f}% | {mean(change):.3f}% |")
    lines.extend(["", "## Paired comparison", ""])
    for i, (condition, label) in enumerate(zip(CONDITIONS[:2], LABELS[:2])):
        seq, joint = panel[condition], panel["joint"]
        n = sum(seq[k][i] > joint[k][i] for k in WRITERS)
        diff = [seq[k][2] - joint[k][2] for k in WRITERS]
        lines.append(f"- {label}: the first-removed direction has a larger remaining projection than joint removal in {n}/{len(WRITERS)} writers. The paired difference in relative weight change (sequential minus joint) ranges from {min(diff):+.3f} to {max(diff):+.3f} percentage points, with mean {mean(diff):+.3f}.")
    lines.extend([
        "",
        "![Writer-level comparison](figures/writer_geometry.svg)",
        "",
        "Each point represents one writer under one condition. Similar weight changes do not establish matched activation distortion or behavioural effects. The writers are parts of a single checkpoint, not independent model replications. The pilot uses regenerated directions and the previously published upstream joint-projection correction.",
        "",
        "Reproduce from the repository root: `python3 analysis/writer_geometry.py`.",
    ])
    return "\n".join(lines) + "\n"


def figure(panel):
    width, height = 760, 420
    left, right, top, bottom = 80, 730, 88, 340
    points = [(c, max(v[:2]), v[2]) for c in CONDITIONS for v in panel[c].values()]
    if any(y <= 0 for _, y, _ in points):
        raise ValueError("The logarithmic plot requires positive projection ratios")
    xmin = math.floor(min(x for _, _, x in points) * 5) / 5
    xmax = math.ceil(max(x for _, _, x in points) * 5) / 5
    if xmin == xmax:
        xmax += .2
    ymin, ymax = .1, 50.
    if any(not ymin <= y <= ymax for _, y, _ in points):
        raise ValueError("Projection ratio outside the plot's fixed range")
    def sx(x):
        return left + (x - xmin) / (xmax - xmin) * (right - left)
    def sy(y):
        return bottom - math.log10(y / ymin) / math.log10(ymax / ymin) * (bottom - top)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Remaining projection versus relative weight change across 53 writers">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        '<g font-family="Arial, sans-serif" fill="#20262d">',
        '<text x="80" y="27" font-size="19" font-weight="bold">Writer-level projection and weight change</text>',
        '<text x="80" y="47" font-size="12">Gemma 2 2B Instruct, archived pilot; one point per writer and condition</text>',
    ]
    for i, (label, color) in enumerate(zip(LABELS, COLORS)):
        x = 82 + i * 160
        svg.append(f'<circle cx="{x}" cy="68" r="4" fill="{color}"/><text x="{x + 10}" y="72" font-size="12">{label}</text>')
    for y in (.1, .3, 1, 3, 10, 30, 50):
        py = sy(y)
        svg.append(f'<path d="M{left} {py:.1f}H{right}" stroke="#e0e4e8"/><text x="{left - 8}" y="{py + 4:.1f}" text-anchor="end" font-size="11">{y:g}</text>')
    for i in range(6):
        x = xmin + i * (xmax - xmin) / 5
        px = sx(x)
        svg.append(f'<path d="M{px:.1f} {top}V{bottom}" stroke="#eef0f2"/><text x="{px:.1f}" y="{bottom + 18}" text-anchor="middle" font-size="11">{x:.2f}</text>')
    for c, color in zip(CONDITIONS, COLORS):
        for k in WRITERS:
            s1, s2, change = panel[c][k]
            svg.append(f'<circle cx="{sx(change):.1f}" cy="{sy(max(s1, s2)):.1f}" r="3.4" fill="{color}" fill-opacity="0.75"/>')
    svg.extend([
        f'<path d="M{left} {top}V{bottom}H{right}" fill="none" stroke="#737c84"/>',
        '<text x="405" y="384" text-anchor="middle" font-size="13">Relative writer weight change (%)</text>',
        '<text transform="translate(20 214) rotate(-90)" text-anchor="middle" font-size="13">Max remaining target projection (%) · log scale</text>',
        '</g></svg>',
    ])
    return "\n".join(svg) + "\n"


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=root / "analysis")
    args = parser.parse_args()
    panel = load_writers(root)
    report, svg = summary(panel), figure(panel)
    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    (args.out / "summary.md").write_text(report, encoding="utf-8")
    (args.out / "figures" / "writer_geometry.svg").write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
