import sys
sys.stdout.reconfigure(encoding='utf-8')
from model import WaferModel

m = WaferModel()
for g in m.groups():
    df = m.stats(g)
    print(f"\n{'='*60}")
    print(f"GROUP: {g}")
    print(f"{'='*60}")
    for _, row in df.iterrows():
        print(f"  [{row['label']}]  n6={row['n_6']}  n8={row['n_8']}")
