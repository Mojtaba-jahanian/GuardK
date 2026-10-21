#!/usr/bin/env python3
"""Validate supplied summary evidence, not the clustering implementation."""
from pathlib import Path
import csv, hashlib, math
ROOT = Path(__file__).resolve().parents[1]
def rows(name):
    with (ROOT / "evidence" / name).open(newline="") as stream:
        return list(csv.DictReader(stream))
def near(a, b, tolerance=1e-10):
    if not math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"Summary inconsistency: {a} != {b}")
def main():
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Integrity mismatch: {name}")
    overall = rows("results__g9_confirmatory__G9_confirmatory_aggregate.csv")
    if len(overall) != 6 or any(int(row["runs"]) != 288 for row in overall):
        raise ValueError("Expected six methods with 288 runs each")
    guard = next(row for row in overall if row["method"] == "guardk_v04")
    near(float(guard["coverage"]) * 288, 161)
    near(float(guard["exact_k_rate_all"]) * 288, 66)
    near(float(guard["exact_k_rate_accepted"]) * 161, 66)
    by_overlap = rows("results__g9_confirmatory__G9_by_overlap.csv")
    levels = [row for row in by_overlap if row["method"] == "guardk_v04"]
    if len(levels) != 3:
        raise ValueError("Expected three overlap strata")
    near(sum(float(row["coverage"]) * 96 for row in levels), 161)
    near(sum(float(row["exact_k_rate_all"]) * 96 for row in levels), 66)
    accepted_ari = sum(float(row["mean_ARI_accepted"]) * float(row["coverage"]) * 96 for row in levels) / 161
    near(accepted_ari, guard["mean_ARI_accepted"])
    ci = rows("results__g9_confirmatory__G9_confirmatory_bootstrap_CI.csv")
    for row in ci:
        if not float(row["ci95_low"]) <= float(row["mean"]) <= float(row["ci95_high"]):
            raise ValueError("Invalid recorded confidence interval")
    print("PASS: input hashes, 288-case counts, retained cardinalities, subgroup aggregation and CI bounds")
    print("Scope: summary-file checks only; no full GuardK experiment was rerun.")
if __name__ == "__main__":
    main()
