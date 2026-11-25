"""Export condition-blind rows or summarize paired human ratings."""
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from pain_axis.geometry import CONDITIONS


def export(run, prompts, out):
    out.mkdir(parents=True, exist_ok=True)
    texts = {row["id"]: row["text"] for row in json.loads(prompts.read_text())}
    expected = set(json.loads((run / "settings.json").read_text())["prompt_ids"])
    items = []
    seen = set()
    for condition in CONDITIONS:
        for path in sorted(run.glob(f"{condition}.[0-9][0-9][0-9].json")):
            batch = json.loads(path.read_text())
            assert batch["condition"] == condition
            for row in batch["rows"]:
                key = condition, row["id"]
                assert key not in seen
                seen.add(key)
                items.append({"condition": condition, "prompt_id": row["id"],
                              "prompt": texts[row["id"]], "response": row["generation"]})
    assert seen == {(condition, prompt) for condition in CONDITIONS for prompt in expected}
    random.Random(1991).shuffle(items)
    with (out / "rating_sheet.csv").open("x", newline="") as visible, \
         (out / "private_key.csv").open("x", newline="") as hidden:
        sheets = (csv.writer(visible), csv.writer(hidden))
        sheets[0].writerow(["anonymous_id", "prompt", "response", "humor", "coherent", "relevant"])
        sheets[1].writerow(["anonymous_id", "prompt_id", "condition"])
        for i, item in enumerate(items):
            anonymous = f"R{i+1:04d}"
            sheets[0].writerow([anonymous, item["prompt"], item["response"], "", "", ""])
            sheets[1].writerow([anonymous, item["prompt_id"], item["condition"]])


def paired(ratings, key):
    with ratings.open() as handle:
        rows = list(csv.DictReader(handle))
    by_alias = {row["anonymous_id"]: row for row in rows}
    assert len(by_alias) == len(rows)
    groups = defaultdict(dict)
    with key.open() as handle:
        keys = list(csv.DictReader(handle))
    assert set(by_alias) == {row["anonymous_id"] for row in keys}
    metrics = ("humor", "coherent", "relevant")
    for row in keys:
        scores = tuple(int(by_alias[row["anonymous_id"]][s]) for s in metrics)
        assert all(x in (0, 1) for x in scores)
        groups[row["condition"]][row["prompt_id"]] = scores
    result = {}
    for first, second in (("baseline", "s1"), ("baseline", "s2"), ("baseline", "s1_then_s2"),
                          ("baseline", "joint"), ("s1_then_s2", "s2_then_s1"), ("s1_then_s2", "joint")):
        a, b = groups[first], groups[second]
        assert a.keys() == b.keys() and a
        contrasts = {}
        for i, name in enumerate(metrics):
            positive = sum(a[p][i] < b[p][i] for p in a)
            negative = sum(a[p][i] > b[p][i] for p in a)
            total = positive + negative
            value = 2 * sum(math.comb(total, j) for j in range(min(positive, negative) + 1)) / 2**total
            contrasts[name] = {"before": sum(v[i] for v in a.values()),
                               "after": sum(v[i] for v in b.values()),
                               "zero_to_one": positive, "one_to_zero": negative, "n": len(a),
                               "delta": (positive - negative) / len(a),
                               "mcnemar_exact_unadjusted_exploratory": min(value, 1)}
        result[f"{first} -> {second}"] = contrasts
    print(json.dumps(result, indent=2))


def main():
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="task", required=True)
    for name, positional in (("export", ("run", "prompts", "out")), ("paired", ("ratings", "key"))):
        command = sub.add_parser(name)
        for field in positional:
            command.add_argument(field, type=Path)
    args = root.parse_args()
    if args.task == "export":
        export(args.run, args.prompts, args.out)
    else:
        paired(args.ratings, args.key)


if __name__ == "__main__":
    main()
