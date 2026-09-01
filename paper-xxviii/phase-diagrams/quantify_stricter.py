import pandas as pd
import numpy as np

df = pd.read_csv("phase_diagram_5D_deterministic.csv")

records = []
for rhoP, group in df.groupby("rhoP"):
    total = len(group)
    weak_bistable = (group["final_regime"] == "bistable").sum()
    
    # Strong bistability: B_open < 0.2 and B_closed > 0.55
    strong = ((group["B_open"] < 0.2) & (group["B_closed"] > 0.55)).sum()
    
    # Both closed
    both_closed = ((group["B_open"] > 0.55) & (group["B_closed"] > 0.55)).sum()
    
    # Both open
    both_open = ((group["B_open"] < 0.2) & (group["B_closed"] < 0.2)).sum()
    
    records.append({
        "rhoP": rhoP,
        "weak_bistable_frac": weak_bistable / total,
        "strong_bistable_frac": strong / total,
        "both_closed_frac": both_closed / total,
        "both_open_frac": both_open / total,
    })

summary = pd.DataFrame(records)
summary.to_csv("phase_diagram_strict_summary.csv", index=False)
print(summary.to_string(index=False))
