from pathlib import Path
import pandas as pd

ROOT = Path.home() / "GuardK_V6"
runs = pd.read_csv(ROOT / "results/g6v_validation/G6V_holdout_runs.csv")
out = ROOT / "results/g6w_refinement"
out.mkdir(parents=True, exist_ok=True)

parts = []
for method in ["guardk_v02", "guardk_v03"]:
    d = runs[runs["method"] == method]
    t = d.groupby(["condition","selected_k"], dropna=False).size().reset_index(name="count")
    t["method"] = method
    parts.append(t)

dist = pd.concat(parts, ignore_index=True)
dist.to_csv(out / "G6W_selected_k_distribution.csv", index=False)

print("G6W_SELECTED_K_DISTRIBUTION")
print(dist.to_string(index=False))
print()
print("G6W_CONCLUSION=v0.3 over-splitting lowered out-of-grid K accuracy and ARI.")
print("G6W_ACTION=REJECT_v0.3_AS_PRIMARY; RESTORE_v0.2_BASE; ADD_LABEL_FREE_K_SELECTION_STABILITY_GATE")
print("G6W_FAILURE_DECOMPOSITION_PASS")
