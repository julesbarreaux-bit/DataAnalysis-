import sys
sys.stdout.reconfigure(encoding='utf-8')
from data import col_limits
from model import WaferModel

print(f"Total columns with limits: {len(col_limits)}")

mdl = WaferModel()
for g in mdl.groups():
    lims = mdl.limits(g)
    if lims:
        print(f"\n=== {g} ===")
        for lbl, lim in lims.items():
            print(f"  [{lbl}]  upper={lim['upper']}  lower={lim['lower']}")

print("\ndone")
