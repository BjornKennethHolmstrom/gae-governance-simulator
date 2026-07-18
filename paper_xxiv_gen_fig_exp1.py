import numpy as np
from scipy.stats import pearsonr
from paper_xxiv_figstyle import apply, PAL
apply()
import matplotlib.pyplot as plt
from paper_xxiv_possibility_exercisable import Grid, collect

env = Grid()
SEEDS = 8
# panel a: passive coupling (epsilon sweep, lambda=0)
passive=[]
for e in (0.1,0.3,0.5,0.7,0.9):
    passive += collect(env,"A",0.0,e,SEEDS,4000)
# panel b: optimization (lambda sweep)
optim=[]
for l in (0.05,0.1,0.2,0.4,0.8):
    optim += collect(env,"B",l,0.15,SEEDS,4000)
# panel c: adaptation — exact 12-seed values from the bootstrap analysis
A=[602,856,880,916,927,967,970,1010,1183,1240,1380,1943]
B=[522,591,694,865,895,934,967,1155,1177,1189,1274,1316]
D=[110,177,225,354,513,624,633,652,711,760,885,1065]

fig,ax=plt.subplots(1,3,figsize=(14.5,4.2))
# (a)
xs=[r["proxy"] for r in passive]; ys=[r["exreach"] for r in passive]; cs=[r["eps"] for r in passive]
sc=ax[0].scatter(xs,ys,c=cs,cmap="viridis",s=30,edgecolor="white",linewidth=0.3)
b,a0=np.polyfit(xs,ys,1); xr=np.linspace(min(xs),max(xs),40)
ax[0].plot(xr,a0+b*xr,"--",color="#333",lw=1.2)
r,_=pearsonr(xs,ys)
ax[0].set_xlabel("biased-resolution proxy (nats)"); ax[0].set_ylabel("exercisable reach")
ax[0].set_title(f"(a) Passive: proxy tracks reach  (r={r:+.2f})")
cbar=fig.colorbar(sc,ax=ax[0]); cbar.set_label("exploration $\\epsilon$")
# (b)
lams=sorted({r["lam"] for r in optim})
agg=lambda k:[np.mean([r[k] for r in optim if r["lam"]==l]) for l in lams]
ax[1].plot(lams,agg("proxy"),"o-",color=PAL["proxy"],lw=1.8,ms=5,label="proxy")
axr=ax[1].twinx(); axr.grid(False)
axr.plot(lams,agg("exreach"),"s--",color=PAL["cost"],lw=1.8,ms=5,label="exercisable reach")
ax[1].set_xlabel("optimization strength  $\\lambda$")
ax[1].set_ylabel("proxy (nats)",color=PAL["proxy"])
axr.set_ylabel("exercisable reach",color=PAL["cost"]); axr.set_ylim(-0.05,1.05)
ax[1].set_title("(b) Optimized: proxy rises, reach stays flat")
# (c)
names=["A\ntask-only","B\nproxy-opt.","D\nstructural"]
mns=[np.mean(A),np.mean(B),np.mean(D)]
ses=[np.std(v)/np.sqrt(len(v)) for v in (A,B,D)]
ax[2].bar(names,mns,yerr=ses,color=[PAL["baseline"],PAL["proxy"],PAL["structural"]],
          width=0.62,error_kw=dict(ecolor="#333",lw=1))
ax[2].set_ylabel("OOD adaptation cost (env. steps)")
ax[2].set_title("(c) Adaptation from common start")
ax[2].annotate("D $-$ A: $-514$ [$-765,-282$]\nB $-$ A: $-108$ [$-353,+110$] (n.s.)",
               xy=(0.5,0.86),xycoords="axes fraction",ha="center",fontsize=8,color="#444")
fig.tight_layout()
fig.savefig("paper_xxiv_fig_exp1_sensor_objective.png")
print("exp1 fig saved; passive r=",round(r,3))
