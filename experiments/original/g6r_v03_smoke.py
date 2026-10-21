import inspect
import numpy as np

from guardk import (
    GuardKV03Config,
    GuardKPrototypeV03,
    make_guardk_scenario,
)

sig = inspect.signature(GuardKPrototypeV03.fit)
assert list(sig.parameters.keys()) == ["self", "X"]

X, y, _ = make_guardk_scenario(
    "balanced_separated",
    2207,
)

cfg = GuardKV03Config(
    seed=2207,
    n_bootstrap=4,
    allow_abstain=True,
)

a = GuardKPrototypeV03(cfg).fit(X)
b = GuardKPrototypeV03(cfg).fit(X)

assert a.selected_k_ == b.selected_k_
assert a.decision_ == b.decision_

np.testing.assert_allclose(
    a.candidate_table_.select_dtypes(include=["number"]).to_numpy(),
    b.candidate_table_.select_dtypes(include=["number"]).to_numpy(),
    rtol=0,
    atol=1e-12,
)

print(a.candidate_table_.to_string(index=False))
print()
print(a.transition_table_.to_string(index=False))
print()
print("selected_k=", a.selected_k_)
print("decision=", a.decision_)

assert a.selected_k_ == 4
assert a.decision_ == "ACCEPT"

print("G6R_V03_API_LABEL_FREE_PASS")
print("G6R_V03_DETERMINISM_PASS")
print("G6R_V03_EASY_SANITY_PASS")
