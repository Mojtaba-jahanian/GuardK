#!/usr/bin/env python3
"""Recompute summary metrics from archived runs; no model fitting."""
from pathlib import Path
import csv, math
R=Path(__file__).resolve().parents[1]
def read(name):
    with (R/name).open(newline="") as f:return list(csv.DictReader(f))
def close(a,b):
    if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10):raise ValueError(f"Mismatch: {a} != {b}")
def main():
    runs=read("archived_results/g9_confirmatory/G9_confirmatory_runs.csv")
    aggregate=read("evidence/results__g9_confirmatory__G9_confirmatory_aggregate.csv")
    for row in aggregate:
        subset=[r for r in runs if r["method"]==row["method"]]
        if len(subset)!=int(row["runs"]):raise ValueError("Run-count mismatch")
        acc=[r for r in subset if float(r["coverage"])==1]
        close(len(acc)/len(subset),float(row["coverage"]))
        exact=sum(float(r["exact_k"]) for r in acc)
        close(exact/len(subset),float(row["exact_k_rate_all"]))
        close(exact/len(acc),float(row["exact_k_rate_accepted"]))
        for input_name,output_name in [("ARI","mean_ARI_accepted"),("NMI","mean_NMI_accepted"),("abs_k_error","mean_abs_k_error_accepted"),("worst_cluster_recall","mean_worst_cluster_recall_accepted")]:
            close(sum(float(r[input_name]) for r in acc)/len(acc),float(row[output_name]))
    print("PASS: all six 288-case method summaries reproduced from 1,728 archived per-run records")
    print("No model was fitted; metrics were reaggregated from archived run-level values.")
if __name__=="__main__":main()
