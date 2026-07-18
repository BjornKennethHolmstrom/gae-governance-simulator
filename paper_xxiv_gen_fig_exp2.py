import json, numpy as np
from paper_xxiv_figstyle import apply, PAL
apply()
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

cache = json.load(open("paper_xxiv_sweep2_cache.json")); MM=13
g=lambda k,cb: cache[f"{k}|{cb}|0.7|{MM}|rowmajor|0"]
KM=(24,32,40,48,56,65); CB=(0.0,0.005,0.01,0.015,0.02)
M=np.array([[np.mean(g(k,cb)["reach"]) for k in KM] for cb in CB])

fig,ax=plt.subplots(1,2,figsize=(11,4.3))
cmap=plt.cm.plasma
for i,cb in enumerate(CB):
    ax[0].plot(KM,M[i],"o-",lw=1.8,ms=5,color=cmap(i/(len(CB)-1)),label=f"$c_b$={cb}")
ax[0].axhline(0.5,color="#999",ls=":",lw=1)
ax[0].set_xlabel("peripheral resolution  $k_{far}$  (categories)")
ax[0].set_ylabel("exercisable reach (optimized proxy)")
ax[0].set_title("(a) Reach vs resolution; boundary shifts with cost")
ax[0].set_ylim(-0.05,1.05); ax[0].legend(title="access cost",ncol=1)

im=ax[1].imshow(M,aspect="auto",origin="lower",cmap="viridis",vmin=0,vmax=1)
ax[1].set_xticks(range(len(KM))); ax[1].set_xticklabels(KM)
ax[1].set_yticks(range(len(CB))); ax[1].set_yticklabels(CB)
ax[1].set_xlabel("peripheral resolution  $k_{far}$")
ax[1].set_ylabel("access cost  $c_b$")
ax[1].set_title("(b) reach$(k_{far}, c_b)$;  ✕ = restoration boundary")
ax[1].grid(False)
for i in range(len(CB)):
    js=[j for j in range(len(KM)) if M[i,j]>=0.9]
    if js: ax[1].plot(min(js),i,"x",color="#d81e5b",ms=11,mew=3)
cb=fig.colorbar(im,ax=ax[1]); cb.set_label("exercisable reach")
fig.tight_layout()
fig.savefig("paper_xxiv_fig_exp2_resolution_cost.png")
print("exp2 fig saved; boundary(reach>=0.9) k* by c_b:",
      [ (CB[i], next((KM[j] for j in range(len(KM)) if M[i,j]>=0.9), '>65')) for i in range(len(CB))])
