import pandas as pd
import numpy as np

df = pd.read_csv("phase_diagram_5D_deterministic.csv")

regime_order = ["open", "intermediate", "closed", "bistable", "oscillatory"]
summary = []

for rhoP, group in df.groupby("rhoP"):
    total = len(group)
    counts = group["final_regime"].value_counts()
    fracs = {regime: counts.get(regime, 0) / total for regime in regime_order}

    theta_vals = sorted(group["theta"].unique())
    collapse_thresholds = []
    recovery_thresholds = []
    hysteresis_widths = []

    for theta in theta_vals:
        sub = group[group["theta"] == theta].sort_values("s").reset_index(drop=True)

        # Collapse: first s where open_regime != "open"
        non_open = sub[sub["open_regime"] != "open"]
        collapse_s = non_open.iloc[0]["s"] if not non_open.empty else np.nan

        # Recovery: highest s where closed_regime is not closed, below the closed block.
        # More robustly, scan from high s downward and find first s where closed_regime != "closed"
        sub_desc = sub.iloc[::-1]
        non_closed = sub_desc[sub_desc["closed_regime"] != "closed"]
        recovery_s = non_closed.iloc[0]["s"] if not non_closed.empty else np.nan

        collapse_thresholds.append(collapse_s)
        recovery_thresholds.append(recovery_s)

        if not np.isnan(collapse_s) and not np.isnan(recovery_s):
            hysteresis_widths.append(collapse_s - recovery_s)

    summary.append({
        "rhoP": rhoP,
        "frac_open": fracs["open"],
        "frac_intermediate": fracs["intermediate"],
        "frac_closed": fracs["closed"],
        "frac_bistable": fracs["bistable"],
        "frac_oscillatory": fracs["oscillatory"],
        "mean_collapse_s": np.nanmean(collapse_thresholds),
        "std_collapse_s": np.nanstd(collapse_thresholds),
        "mean_recovery_s": np.nanmean(recovery_thresholds),
        "std_recovery_s": np.nanstd(recovery_thresholds),
        "mean_hysteresis_width": np.nanmean(hysteresis_widths),
        "std_hysteresis_width": np.nanstd(hysteresis_widths),
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv("phase_diagram_summary_corrected.csv", index=False)
print(summary_df.to_string(index=False))
