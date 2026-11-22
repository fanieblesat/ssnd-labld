"""Selection rule fixed before the tuning runs: for each method, the setting with the smallest mean
certified gap (100 (UB - LB) / UB, own bounds) over all tuning instances wins; a setting must have run on
every tuning instance. Ties within 0.01 percentage points go to the default setting.

    python select_tuning.py results/tune/snd.csv [more csv files from the other machine]
"""
import csv
import sys
from collections import defaultdict

rows = [r for f in sys.argv[1:] for r in csv.DictReader(open(f))]
insts = sorted({r["instance"] for r in rows})
gap = defaultdict(dict)
for r in rows:
    if r["gap_pct"] != "":
        gap[r["method"]][r["instance"]] = float(r["gap_pct"])
print(f"{len(insts)} tuning instances")
for meth in ("ld", "bd"):
    cands = {m: g for m, g in gap.items() if m.startswith(meth + "_")}
    table = []
    for m, g in sorted(cands.items()):
        if set(g) != set(insts):
            print(f"  {m}: incomplete ({len(g)}/{len(insts)} instances), excluded")
            continue
        table.append((sum(g.values()) / len(g), m))
    if not table:
        continue
    best = min(table)
    default = next((t for t in table if t[1] == f"{meth}_def"), None)
    choice = default if default and default[0] - best[0] <= 0.01 else best
    for mean, m in sorted(table):
        print(f"  {m:12s} mean gap {mean:7.3f}%" + ("   <- selected" if m == choice[1] else ""))
