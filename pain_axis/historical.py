"""Read upstream verification data without mixing it with new measurements."""
import argparse
import csv
import json
from pathlib import Path

from pain_axis.geometry import file_sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    args = p.parse_args()
    out = args.source / "results/appC_ablation"
    csv_path = out / "verify_Gemma_2_2B_instruct.csv"
    generations = out / "by_prompt_Gemma_2_2B_instruct.json"
    with csv_path.open() as handle:
        records = list(csv.DictReader(handle))
    fixed = {}
    for record in records:
        if record["kind"] == "strict_fixed" and record["probe"] in ("s1@L10", "s2@L15"):
            fixed.setdefault(record["condition"], {})[record["probe"]] = float(record["mean_abs_proj"])
    data = json.loads(generations.read_text())
    assert len(data) == 100
    conditions = ("baseline", "s1", "s2", "s1s2")
    assert all(set(conditions) <= set(row) for row in data.values())
    print(json.dumps({"source": "upstream outputs, read only",
                      "verification_csv_sha256": file_sha(csv_path),
                      "generation_json_sha256": file_sha(generations),
                      "fixed_projection": fixed, "prompts": len(data),
                      "exact_text_match_to_baseline": {
                          c: sum(row[c] == row["baseline"] for row in data.values())
                          for c in conditions},
                      "paper_reported_counts_not_rescored": {
                          "baseline": 0, "s1": 6, "s2": 17, "s1s2": 26}}, indent=2))


if __name__ == "__main__":
    main()
