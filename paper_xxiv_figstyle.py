import matplotlib as mpl
mpl.use("Agg")

def apply():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.edgecolor": "#555", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.7,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "xtick.color": "#333",
        "ytick.color": "#333", "legend.fontsize": 8.5, "legend.frameon": False,
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
        "figure.facecolor": "white",
    })

PAL = {
    "baseline": "#6b7280",   # task-only / gray
    "proxy":    "#b4433a",   # optimized proxy / red
    "structural":"#2e8b57",  # structural / green
    "mixed":    "#7d5ba6",   # mixed / purple
    "cost":     "#2471a3",   # truth / blue
    "accent":   "#c98a00",   # highlight / amber
}
