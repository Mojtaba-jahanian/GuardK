import inspect
from guardk import GuardKV04Config, GuardKPrototypeV04, make_guardk_scenario

assert list(inspect.signature(GuardKPrototypeV04.fit).parameters.keys()) == ["self","X"]
X, _, _ = make_guardk_scenario("balanced_separated", 2207)
cfg = GuardKV04Config(seed=2207, outer_repeats=5)

a = GuardKPrototypeV04(cfg).fit(X)
b = GuardKPrototypeV04(cfg).fit(X)

print("A_DIAGNOSTICS=", a.diagnostics_)
print("B_DIAGNOSTICS=", b.diagnostics_)
assert a.diagnostics_ == b.diagnostics_
assert a.selected_k_ == 4
assert a.decision_ == "ACCEPT"

print("G6W_V04_API_LABEL_FREE_PASS")
print("G6W_V04_DETERMINISM_PASS")
print("G6W_V04_EASY_SANITY_PASS")
