import json, numpy as np
from scipy.optimize import minimize

cache = json.load(open("./outputs/paper_xxiv_sweep2_cache.json"))
MM = 13; KM = (24,32,40,48,56,65); CB = (0.0,0.005,0.01,0.015,0.02)
def get(k,cb,lam=0.7,part="rowmajor",ps=0): return cache.get(f"{k}|{cb}|{lam}|{MM}|{part}|{ps}")

# ---- (B) normalized entropy: is 'proxy inflation' just a bigger ceiling? ----
print("NORMALIZED proxy  H_norm = H_raw / log(13+k_far)   (raw | Hmax | Hnorm), c_b=0 row:")
for k in KM:
    d = get(k,0.0); raw = np.mean(d["proxy"]); hmax = np.log(MM+k)
    print(f"  k={k:>3}  raw={raw:.2f}  Hmax={hmax:.2f}  Hnorm={raw/hmax:.3f}")

# ---- (A) interaction GLM: binomial reach ~ log2(k) + c_b + log2(k):c_b ----
X, succ = [], []
for cb in CB:
    for k in KM:
        d = get(k,cb)
        for r in d["reach"]:
            X.append([1.0, np.log2(k), cb, np.log2(k)*cb]); succ.append(r)  # r = fraction over 13
X = np.array(X); succ = np.array(succ); n = 13
# center predictors for interpretable interaction
Xc = X.copy()
Xc[:,1] -= X[:,1].mean(); Xc[:,2] -= X[:,2].mean()
Xc[:,3] = Xc[:,1]*Xc[:,2]

def nll(b):
    z = Xc@b; p = 1/(1+np.exp(-z)); p = np.clip(p,1e-9,1-1e-9)
    return -np.sum(n*(succ*np.log(p)+(1-succ)*np.log(1-p)))
res = minimize(nll, np.zeros(4), method="BFGS")
b = res.x
# bootstrap SEs by resampling the 6 seeds within each cell
rng = np.random.default_rng(0); boots=[]
cellmap = {}
idx=0
for ci,cb in enumerate(CB):
    for k in KM:
        cellmap[(k,cb)] = list(range(idx, idx+6)); idx+=6
for _ in range(400):
    rows=[]
    for key,ids in cellmap.items():
        rows += list(rng.choice(ids,6))
    Xb=Xc[rows]; sb=succ[rows]
    def nllb(bb):
        z=Xb@bb; p=np.clip(1/(1+np.exp(-z)),1e-9,1-1e-9)
        return -np.sum(n*(sb*np.log(p)+(1-sb)*np.log(1-p)))
    rb=minimize(nllb,b,method="BFGS")
    boots.append(rb.x)
boots=np.array(boots)
names=["intercept","log2(k_far)","c_b","log2(k_far):c_b"]
print("\nINTERACTION GLM (binomial, seed-level fractions; predicted b1>0, b2<0, b3<0):")
for i,nm in enumerate(names):
    lo,hi=np.percentile(boots[:,i],[2.5,97.5])
    print(f"  {nm:>16} = {b[i]:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]")
