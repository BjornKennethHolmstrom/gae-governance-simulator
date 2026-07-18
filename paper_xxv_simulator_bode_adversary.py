"""
GaE engineering cycle -- Bode sensitivity under strategic disturbance allocation
Working title: "Where Reform Pushes Down, Strategy Pushes Back"
seed: 20260716   (round-3 corrections)

SPINE: an IMPORTED allocation lemma, plus Bode-achievability in a different geometry.

  IMPORTED  allocation lemma [imported: CVaR/Expected-Shortfall risk envelope;
            equivalently DRO with a likelihood-ratio-bounded ambiguity set]
      Over densities q on a base measure mu_a with 0<=q<=c, int q dmu_a=1:
        sup_q int f q dmu_a = CVaR_{1-1/c}(f ; mu_a),   f=|S|^2.
      So J_strat/E = CVaR_{1-1/c}(f | mu_a on Omega_a); J_pass/E = E_{mu_a}[f];
      G/E = CVaR - mean = M*U (M=mean, U=CVaR/mean-1). c=1->mean; c->inf->ess sup.
      NOT novel (Rockafellar-Uryasev). Stated for a general base measure mu_a
      (uniform is a special case). Caveat: STATIC lemma -- loss linear in density,
      no temporal/causal/transition/discovery cost (see DISCOVERY below).

  BODE-ACHIEVABILITY [R] -- where any originality lives.
      B_S = int_0^inf ln|S| dw = pi*sum Re(p_k of L, RHP) >= 0; =0 here.
      So A_+ = A_- >= A_m for every proper, internally stabilizing controller
      with no prohibited unstable cancellations. Bode constrains [ln|S|]_+; the
      adversary applies CVaR to |S|^2 -- different geometries, so accessible
      log-area does NOT determine loss (functional non-identifiability demo below).
      The open technical object is the FEASIBLE IMAGE
        {(A_m, A_+, CVaR_a(|S_K|^2|Om_a), M_s, ||W_u KS||, ||W_T T||_inf): K admissible}
      and its geometry (inf CVaR s.t. A_m>=Abar; Pareto front). Synthesis is an
      ACHIEVABILITY experiment, not a re-test of the imported lemma. The one narrow
      novelty candidate: design against the intermediate CVaR functional -- a
      continuous interpolation between band-limited H2 (c=1) and finite-frequency
      H_inf (c->inf, generalized KYP endpoint) indexed by adversary capability.
      The inner adversary is CVaR (closed form) and |S(Q)|^2 is convex in the Youla
      Q, so inf_Q CVaR is a CONVEX program (Rockafellar-Uryasev linearization) ->
      the minimax is solvable, not a sweep.

  DISCOVERY [open licensing capacity]. CVaR is realized only by an adversary that
      KNOWS f (can locate the upper tail). Reach (Omega_a), concentration (c) and
      discovery (legibility of the peak) are three separate capacities; the model
      grants the third free. Noisy knowledge of f realizes strictly less than CVaR.

  R0  B_S -> 0 as the analytic tail: ln|S| = 1/(2w^2)+693/(8w^4)+O(w^-6) =>
      B_S(0,W) = -1/(2W) - 231/(8W^3)+O(W^-5); residual B_S+kp/W ~ -28.875/W^3
      (matches the run). R4a: G>0 <=> c>1 and f nonconstant mu_a-a.e. (E[f]>0).
      R4b: J_strat>E <=> CVaR_a(f|Om_a)>1.
      R4c relocation [R/open]: CT premise is FALSE for this class -- rel deg >=2
      forces |S|>1 on an unbounded high-freq tail (area ~ kp/w_c beyond any cutoff),
      so amplification cannot be confined to a finite certified band. State the peak
      bound only with the tail term, or in discrete time. But DT closes only export
      beyond Nyquist, NOT relocation outside Omega_a or into weakly-penalized/near-
      Nyquist bands; the |T|-peak/HF constraint stays indispensable.

  Governance [IP]: Layer A (allocation) transfers without Bode; Layer B (migration:
      reform MANUFACTURES the peak) needs an empirically conserved burden + domain +
      relocation mechanism + model boundary. High loss needs no adversary: under a
      colored baseline q0, E_{q0}[f] can be high even when the uniform mean is <1.
"""
import numpy as np
SEED=20260716
kp,wm,Q=0.5,3.0,2.0
P=lambda jw:1.0/((jw+1.0)*(jw+2.0)); K=lambda jw,d:kp+d*jw/(jw**2+(wm/Q)*jw+wm**2)
Sf=lambda jw,d:1.0/(1.0+P(jw)*K(jw,d))
w=np.linspace(0.0,300.0,400_000); jw=1j*w; dw=w[1]-w[0]
Omega_m=(w>wm-0.6)&(w<wm+0.6); E=1.0

def areas(Sv):
    l=np.log(np.abs(Sv))
    return (np.trapezoid(np.clip(l,0,None),w),-np.trapezoid(np.clip(l,None,0),w),
            -np.trapezoid(np.clip(l[Omega_m],None,0),w[Omega_m]))

# --- TWO INDEPENDENT implementations of strategic loss (restored cross-check) ---
def strat_greedy(Sv,mask,c):
    """Operational: water-fill density D_max=c*E/W onto highest-|S|^2 cells until E spent."""
    f=np.abs(Sv[mask])**2; W=f.size*dw; Dmax=c*E/W
    o=np.argsort(f)[::-1]; D=np.zeros_like(f); rem=E; cell=Dmax*dw
    for i in o:
        if rem<=0: break
        t=min(cell,rem); D[i]=t/dw; rem-=t
    return float(np.sum(f*D*dw))
def strat_cvar(Sv,mask,c):
    """Analytic: E * upper-tail mean (CVaR) over top 1/c measure, fractional boundary."""
    f=np.abs(Sv[mask])**2; W=f.size*dw; mcov=W/c
    v=np.sort(f)[::-1]; cum=np.arange(1,f.size+1)*dw; k=int(np.searchsorted(cum,mcov))
    if k>=f.size: cvar=f.mean()
    else:
        covered=cum[k-1] if k>0 else 0.0
        cvar=(v[:k].sum()*dw+v[k]*(mcov-covered))/mcov
    return E*cvar
def passive(Sv,mask): return E*float(np.mean(np.abs(Sv[mask])**2))

print(f"seed={SEED}")
# functional non-identifiability (arbitrary profiles, NOT achievable S; points at synthesis)
def demo(vals,meas,c=2.0):
    Ap=np.sum(np.maximum(0.5*np.log(vals),0)*meas); o=np.argsort(vals)[::-1]
    v=vals[o]; m=meas[o]; mc=meas.sum()/c; acc=num=0
    for vi,mi in zip(v,m):
        t=min(mi,mc-acc); num+=vi*t; acc+=t
        if acc>=mc: break
    return Ap,num/mc
a=demo(np.array([4.,1.]),np.array([.25,.75])); b=demo(np.array([2.,1.]),np.array([.5,.5]))
print(f"[functional non-identifiability] equal A_+={a[0]:.4f}={b[0]:.4f}, J_strat/E {a[1]:.2f} vs {b[1]:.2f}"
      f"  (arbitrary profiles, not achievable S -- synthesis tests the achievable manifold)")

print("\n[R0] B_S+kp/W vs analytic -28.875/W^3:")
for W in (100,300,1000,3000):
    N=int(W/7.5e-4); ww=np.linspace(0.,W,N); B=np.trapezoid(np.log(np.abs(Sf(1j*ww,20.0))),ww)
    print(f"  W={W:5}: B_S+kp/W={B+kp/W: .3e}   -28.875/W^3={-28.875/W**3: .3e}")

# R2: c-sweep with greedy-vs-cvar cross-check (the restored verification)
print("\n[R2] c-sweep (delta=20, band[0.1,20]); |greedy - CVaR| is the cross-check")
Sv=Sf(jw,20.0); full=(w>0.1)&(w<20); Jp=passive(Sv,full)
print("  {:>4} {:>8} {:>9} {:>9} {:>10}".format("c","Jp/E","Jstr/E","G/E","|g-cvar|"))
for c in (1,2,5,20):
    Jg=strat_greedy(Sv,full,c); Jc=strat_cvar(Sv,full,c)
    print("  {:4.1f} {:8.4f} {:9.4f} {:9.4f} {:10.2e}".format(c,Jp/E,Jc/E,(Jc-Jp)/E,abs(Jg-Jc)))

# R3: sliding window, greedy-vs-cvar cross-check; report M,U as decomposition (not a check)
print("\n[R3] sliding window (c=2,width=2); A_+=%.3f fixed" % areas(Sv)[0])
pk=w[np.argmax(np.abs(Sv))]; Wa=2.0
print("  {:>4} {:>8} {:>8} {:>8} {:>7} {:>7} {:>10}".format("z","Jp/E","Jstr/E","G/E","M","U","|g-cvar|"))
zs=np.arange(0.5,8.01,0.5); rows=[]
for z in zs:
    m=(w>=z)&(w<z+Wa); Jp=passive(Sv,m); Jg=strat_greedy(Sv,m,2.0); Jc=strat_cvar(Sv,m,2.0)
    f=np.abs(Sv[m])**2; M=f.mean(); U=(Jc/E)/M-1; rows.append((z,Jp,Jc))
    if z in (0.5,2.0,3.5,6.0,8.0):
        print("  {:4.1f} {:8.4f} {:8.4f} {:8.4f} {:7.3f} {:7.3f} {:10.2e}".format(
              z,Jp/E,Jc/E,(Jc-Jp)/E,M,U,abs(Jg-Jc)))
rows=np.array(rows); G=rows[:,2]-rows[:,1]
m0=(w>=0.5)&(w<2.5); print(f"  [R4a] z=0.5: max|S|^2={np.abs(Sv[m0]).max()**2:.3f}<1 yet G/E={(strat_cvar(Sv,m0,2.0)-passive(Sv,m0))/E:.4f}>0")

# margin erosion: the proxy optimizer spends robustness (Bode->M_s->GM,PM)
print("\n[margins] proxy suppression spends stability margins (M_s=peak|S|):")
print("  {:>5} {:>6} {:>8} {:>8}".format("delta","M_s","GM(dB)","PM(deg)"))
for d in (0,8,14,20):
    Ms=np.abs(Sf(jw,d)).max(); GM=20*np.log10(Ms/(Ms-1)); PM=np.degrees(2*np.arcsin(1/(2*Ms)))
    print("  {:5.0f} {:6.2f} {:8.2f} {:8.1f}".format(d,Ms,GM,PM))

# figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, os
os.makedirs("outputs",exist_ok=True)
JpE=rows[:,1]/E; JsE=rows[:,2]/E; RG=rows[:,2]/rows[:,1]
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
ax[0].axvspan(pk-Wa,pk,color="#48c",alpha=.10,label="peak inside window")
ax[0].plot(zs,JpE,"s-",color="#3b7",label="$J_{pass}/E$")
ax[0].plot(zs,JsE,"o-",color="#c33",label="$J_{strat}/E=\\mathrm{CVaR}_\\alpha$")
ax[0].axhline(1,ls=":",color="#999",lw=.8); ax[0].set_ylabel("normalized loss")
ax[0].set_xlabel("window left edge $z$"); ax[0].set_title("accessibility (global $A_+{=}1.52$ fixed)"); ax[0].legend(fontsize=8)
ax[1].plot(zs,G/E,"o-",color="#c33",label="$G/E=M\\!\\cdot\\!U$ (absolute)")
ax[1].plot(zs,RG-1,"s-",color="#e8a33d",label="$U=R_G-1$ (relative)")
ax[1].set_ylabel("normalized / relative premium"); ax[1].set_xlabel("window left edge $z$")
ax[1].set_title("premium decomposition"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig("outputs/paper_xxv_bode-adversary-boundaries.png",dpi=130)
print("\nsaved outputs/paper_xxv_bode-adversary-boundaries.png")

# mechanism figure (folded in for repo self-consistency; supersedes delta-sweep.png,
# which was built on the retired absolute-cap adversary)
fig,ax=plt.subplots(figsize=(8,4.2))
ax.axvspan(wm-0.6,wm+0.6,color="#48c",alpha=.12,label="monitored band $\\Omega_m$")
for d,col in [(2,"#8bc"),(10,"#e8a33d"),(20,"#c0392b")]:
    ax.plot(w,np.abs(Sf(jw,d)),color=col,lw=1.6,label=f"$|S|$, $\\delta={d}$")
ax.axhline(1,color="#999",ls="--",lw=.8); ax.set_xscale("log"); ax.set_xlim(.1,300); ax.set_ylim(0,3.6)
ax.set_xlabel("$\\omega$"); ax.set_ylabel("$|S|$"); ax.legend(fontsize=8,loc="upper left")
ax.set_title("Proxy suppression manufactures the peak (Bode-conserved)")
fig.tight_layout(); fig.savefig("outputs/paper_xxv_bode-adversary-mechanism.png",dpi=130)
print("saved outputs/paper_xxv_bode-adversary-mechanism.png")
