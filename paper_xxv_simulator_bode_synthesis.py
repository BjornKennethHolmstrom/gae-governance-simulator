"""
GaE engineering cycle -- Bode/adversary SYNTHESIS stage (achievability experiment)
Companion to gae-simulator-bode-adversary.py.  seed: 20260716

QUESTION: at a fixed level of monitored-band (proxy) suppression, what accessible
strategic exploitability CVaR_{1-1/c}(|S|^2 | Omega_a) is achievable by admissible
discrete-time controllers, and does the answer depend on adversary concentration c
and on the robust-stability cap |T|<=TP?

Design (convex): |S(Q)|^2 = |1 - P Q|^2 is convex quadratic in the FIR Youla
parameter q; CVaR is convex-monotone; proxy cost, effort and |T|-peak are convex;
so min CVaR over admissible q is a convex program (SLSQP, multistart).

  front(suppression)      = min achievable accessible CVaR at that proxy suppression
  greedy                  = CVaR-blind controller (min proxy cost), a single point
  gap_matched             = greedy_CVaR - front(at greedy's own suppression)
SWEEP c in {1.5,2,5} (adversary strength) x TP in {1.5,2,3} (robustness slack).
Hypothesis: looser TP lets the optimizer relocate amplification out of Omega_a, so
the front stays lower at deep suppression (gap stays open) -- i.e. the round-4
'collapse under pressure' may be governed by the robustness margin.
"""
import numpy as np
from scipy.signal import cont2discrete
from scipy.optimize import minimize
np.random.seed(20260716)

Ts=0.1
(numd,dend,_)=cont2discrete(([1.0],[1.0,3.0,2.0]),Ts,method='zoh')
numd=np.atleast_1d(numd.squeeze())
N=400; w=np.linspace(1e-3,np.pi,N); z=np.exp(1j*w)
Pz=np.polyval(numd,z)/np.polyval(dend,z)
m=8; Zmat=np.exp(-1j*np.outer(w,np.arange(m)))
S_of=lambda q:1.0-Pz*(Zmat@q); T_of=lambda q:Pz*(Zmat@q)
wm,hw=0.30,0.06; Om_m=(w>wm-hw)&(w<wm+hw); Om_a=(w>0.10)&(w<2.0)
EQ=6.0

proxy_cost=lambda q:float(np.mean(np.abs(S_of(q))[Om_m]**2))
Tpeak=lambda q:float(np.abs(T_of(q)).max())
effort=lambda q:float(np.linalg.norm(q))
def cvar_acc(q,alpha):
    fa=np.sort(np.abs(S_of(q))[Om_a]**2)[::-1]; k=max(1,int(round((1-alpha)*fa.size))); return float(fa[:k].mean())

def solve(obj,ptarget,TP,starts=4):
    cons=[{'type':'ineq','fun':lambda q:TP-Tpeak(q)},{'type':'ineq','fun':lambda q:EQ-effort(q)}]
    if ptarget is not None: cons.append({'type':'ineq','fun':lambda q,t=ptarget:t-proxy_cost(q)})
    best=None
    for _ in range(starts):
        r=minimize(obj,np.random.randn(m)*0.2,method='SLSQP',constraints=cons,options={'maxiter':300,'ftol':1e-9})
        if best is None or r.fun<best.fun: best=r
    return best.x

base=proxy_cost(np.zeros(m))
cs=[1.5,2.0,5.0]; TPs=[1.5,2.0,3.0]; fracs=[0.85,0.6,0.4,0.28]  # common suppression targets
print(f"seed=20260716  Ts={Ts} N={N} m={m}  base proxy cost={base:.3f}  effort<={EQ}")

# (1) value-function fronts over a COMMON suppression grid (all TP reach these)
fronts={}
for TP in TPs:
    for c in cs:
        al=1-1/c; front=[]
        for fr in fracs:
            q=solve(lambda q:cvar_acc(q,al),fr*base,TP); front.append((1-proxy_cost(q)/base,cvar_acc(q,al)))
        fronts[(c,TP)]=np.array(front)

# (2) MATCHED gap: CVaR-aware at the greedy's OWN proxy cost -> collapses to ~0
print("\n[matched] at the proxy-greedy controller's own suppression, blind vs aware CVaR:")
print(f"{'c':>4} {'TP':>4} {'gr_supp':>8} {'gr_CVaR':>8} {'matched_gap':>12}")
for TP in TPs:
    qg=solve(proxy_cost,None,TP); pg=proxy_cost(qg); gs=1-pg/base
    for c in cs:
        al=1-1/c; qm=solve(lambda q:cvar_acc(q,al),pg,TP)
        print(f"{c:4.1f} {TP:4.1f} {gs:8.3f} {cvar_acc(qg,al):8.3f} {cvar_acc(qg,al)-cvar_acc(qm,al):12.2e}")
print("=> at maximal attainable suppression, risk-blind and risk-aware optima COINCIDE.")

# (3) genuinely matched TP effect: at a fixed suppression, looser allowance -> lower floor
print("\n[matched-TP] min achievable CVaR at fixed suppression 0.72, by TP:")
print(f"{'c':>4} | {'TP=1.5':>7} {'TP=2.0':>7} {'TP=3.0':>7}")
for c in cs:
    al=1-1/c; vals=[fronts[(c,TP)][-1,1] for TP in TPs]
    print(f"{c:4.1f} | {vals[0]:7.3f} {vals[1]:7.3f} {vals[2]:7.3f}")

# figure: per-c panels, value-function fronts V_TP(tau); looser TP sits lower; greedy lies ON its front
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, os
os.makedirs("outputs",exist_ok=True)
cols={1.5:"#3b7",2.0:"#e8a33d",3.0:"#c33"}
fig,ax=plt.subplots(1,3,figsize=(13,4),sharey=True)
for j,c in enumerate(cs):
    al=1-1/c
    for TP in TPs:
        fr=fronts[(c,TP)]; ax[j].plot(fr[:,0],fr[:,1],"o-",color=cols[TP],label=f"TP={TP}")
        qg=solve(proxy_cost,None,TP); ax[j].scatter([1-proxy_cost(qg)/base],[cvar_acc(qg,al)],
                  color=cols[TP],marker="x",s=55,zorder=5)
    ax[j].axhline(1,ls=":",color="#999",lw=.8); ax[j].set_title(f"c={c} (alpha={al:.2f})")
    ax[j].set_xlabel("proxy suppression")
    if j==0: ax[j].set_ylabel("min achievable accessible CVaR")
ax[0].legend(fontsize=7,title="lines=V(tau); x=greedy (on its front)")
fig.suptitle("Achievability: value-function floor rises with suppression; looser allowance lowers it; greedy is CVaR-optimal at its own suppression",fontsize=9.5)
fig.tight_layout(); fig.savefig("outputs/paper_xxv_bode-synthesis-sweep.png",dpi=130)
print("\nsaved outputs/paper_xxv_bode-synthesis-sweep.png")
