import numpy as np
from paper_xxiv_figstyle import apply, PAL
apply()
import matplotlib.pyplot as plt
from paper_xxiv_possibility_future_3 import run

env, rows = run(K=78, reach_seeds=6)   # rows: (reg, eff_mead, eff_far, pshare, reach_mean, reach_std)
labels={"task_only":"task-only","passive":"passive\nexploration","structural":"structural\nexposure","mixed":"mixed"}
cols=[PAL["proxy"],PAL["accent"],PAL["structural"],PAL["mixed"]]
regs=[labels[r[0]] for r in rows]; efff=[r[2] for r in rows]
share=[r[3] for r in rows]; rea=[r[4] for r in rows]; er=[r[5] for r in rows]

fig,ax=plt.subplots(1,2,figsize=(11,4.3))
b=ax[0].bar(regs,efff,color=cols,width=0.62)
ax[0].axhline(40,color="#999",ls="--",lw=1.2,label="Exp. 2 restoration\nthreshold (~40)")
ax[0].set_ylabel("learned peripheral resolution\n(distinct categories, of 78)")
ax[0].set_title("(a) How each regime allocates categories to the periphery")
ax[0].legend(loc="upper left")
for r,s in zip(b,share):
    ax[0].text(r.get_x()+r.get_width()/2, r.get_height()+1.2,
               f"{s*100:.1f}%\nvisits", ha="center",va="bottom",fontsize=8,color="#555")
ax[0].set_ylim(0,52)

b2=ax[1].bar(regs,rea,yerr=er,color=cols,width=0.62,error_kw=dict(ecolor="#333",lw=1))
ax[1].set_ylabel("exercisable reach (optimizing the learned proxy)")
ax[1].set_ylim(0,1.08)
ax[1].set_title("(b) Does optimizing the learned proxy decouple?")
fig.tight_layout()
fig.savefig("paper_xxiv_fig_exp3_learned_allocation.png")
print("exp3 fig saved")
