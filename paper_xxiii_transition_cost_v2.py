#!/usr/bin/env python3
"""
paper_xxiii_transition_cost_v2.py
=================================
v2.  The cost metric is rebuilt.  v1's was invalid.

WHAT WAS WRONG WITH v1
----------------------
v1 defined d_cost(A->B) as the excess error, while fine-tuning A on B's home regime,
above **B's own converged loss**.  That makes the floor a property of the TARGET
MODEL's capacity, not of the target REGIME.  So if A is simply a better model than B,
A already beats B's floor before a single gradient step and the transition costs 0.

The smoke test showed the damage immediately: median relative asymmetry came out at
exactly 1.000, the signature of a bug rather than a phenomenon.  compressed_h2 (h=2,
a weak model with a lax floor) cost ~11-12 to leave and exactly 0.00 to enter, from
anywhere.  The "0.787 cost asymmetry" was capacity, not hysteresis, and it contaminated
Test 4's rho as well.  Neither test measured what it claimed to.

THE FIX
-------
The floor is now a property of the target REGIME and of A's OWN architecture:

    floor(A, R) = converged loss of a FRESHLY TRAINED model with A's architecture,
                  trained from scratch on regime R.

    d_cost(A -> R) = integral over the fine-tuning budget of max(0, loss_t - floor(A,R))

This asks the question that actually matters, and it is a governance question:

    How much worse is it to REFORM an existing institution into a fit for regime R
    than to BUILD A NEW ONE, of the same capacity, for R directly?

Cost is now capacity-fair.  It is zero when fine-tuning matches a fresh build at once
(perfect transfer), and the SIGNED variant goes negative under positive transfer --
when the reformed institution is better than a purpose-built one, because it carried
something useful with it.  That quantity did not exist in v1 and it is worth having.

Reference floors: 4 architectures x 4 regimes = 16, of which 7 are already in the zoo
(each zoo model IS a fresh model of its architecture on its home regime).  So 9 extra
trainings per seed.

Also fixed: `float(loss)` on a graph-attached tensor (the UserWarning), and the loss
tracked during fine-tuning is now a periodic full-dataset evaluation rather than a
noisy single-batch figure.

REGISTERED PREDICTIONS (unchanged in substance; the measurement is what changed)
-------------------------------------------------------------------------------
  H_predict  : Spearman rho(d_beh, d_cost) >= RHO_BAR, median across seeds, and
               >= RHO_FLOOR in >= PASS_BAR seeds.
               NULL: the geometry is DECORATIVE -- it does not tell an institution
               what a reform will cost -- and Paper XXIII should not be written.

  H_geodesic : at least one (A,B) pair for which a chained detour A->C->B beats the
               direct move by >= DOMINANCE_MARGIN, in >= PASS_BAR seeds.
               NULL: the direct transition is never dominated; "geodesic reform" is
               a metaphor.

  Reported regardless: the SYMMETRIC-PREDICTOR CEILING -- the correlation d_beh
  achieves against the symmetrized cost.  Behavioral distance is symmetric by
  construction; if cost is not, then no symmetric quantity can predict it fully, and
  the ceiling tells us whether d_beh is failing on its own terms or hitting a
  structural bound.  If the object is genuinely asymmetric, factorization space is
  not a metric space and the Riemannian framing was wrong from the start -- which
  exploration 04 already warned about, and which is a finding, not an embarrassment.

Runtime: ~2-3.5 h for N_SEEDS=10 on 8 cores.  N_SEEDS=2 to smoke-test.
"""

import os, itertools, copy
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from scipy.stats import spearmanr

from paper_xxiii_geometry_replication_v2 import (
    MODEL_SPECS, MODEL_NAMES, N_MODELS,
    rollout, schedule_stream, windows, train_arch, train_model,
    predict_stream, dmat_from_preds, OUT,
)

# ======================================================================
# REGISTERED CONSTANTS
# ======================================================================
N_SEEDS          = 2 # 10
PASS_BAR         = 2 # 8
RHO_BAR          = 0.50
RHO_FLOOR        = 0.30
FT_BUDGET        = 400
FT_LR            = 5e-4
FT_BATCH         = 128
EVAL_EVERY       = 20      # full-dataset evaluation cadence during fine-tuning
N_COSTLIEST      = 6
DOMINANCE_MARGIN = 0.10

torch.set_num_threads(os.cpu_count() or 8)

HOME  = {n: s[2] for n, s in zip(MODEL_NAMES, MODEL_SPECS)}          # model -> home regime
ARCH  = {n: (s[1], s[3]) for n, s in zip(MODEL_NAMES, MODEL_SPECS)}  # model -> (hidden, kind)
KIND  = {n: s[3] for n, s in zip(MODEL_NAMES, MODEL_SPECS)}
REGIMES = ['normal', 'wind', 'damped', 'blur']


def make_dataset(regime, rng, n_episodes=16, ep_len=600):
    Xs, Yps, Yvs = [], [], []
    for _ in range(n_episodes):
        F, P = rollout(regime, ep_len, rng)
        x, yp, yv = windows(F, P); Xs.append(x); Yps.append(yp); Yvs.append(yv)
    return torch.cat(Xs), torch.cat(Yps), torch.cat(Yvs)


@torch.no_grad()
def eval_loss(model, X, Yp, batch=512):
    """Position MSE on the full dataset.  Detached -- no graph, no warning."""
    was = model.training; model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(X), batch):
        o = model(X[i:i+batch])
        pos = o[0] if isinstance(o, tuple) else o
        tot += float(nn.functional.mse_loss(pos, Yp[i:i+batch], reduction='sum').item())
        n += pos.numel()
    if was: model.train()
    return tot/n


def build_floors(seed, data):
    """floor[(arch, regime)] = converged loss of a fresh model of that architecture,
    trained from scratch on that regime.  Capacity-fair by construction.
    Seven of the sixteen are already in the zoo; only nine need training."""
    rng = np.random.default_rng(4000+seed)
    archs = sorted(set(ARCH.values()))
    floors, refs = {}, {}
    for (hid, kind) in archs:
        for R in REGIMES:
            native = next((n for n in MODEL_NAMES
                           if ARCH[n] == (hid, kind) and HOME[n] == R), None)
            refs[((hid, kind), R)] = native      # None => must train fresh
    for key, native in refs.items():
        (hid, kind), R = key
        if native is None:
            m = train_arch(hid, kind, R, rng)
            floors[key] = eval_loss(m, data[R][0], data[R][1])
    return floors, refs


def finetune(model, kind, data, floor, budget=FT_BUDGET):
    """Fine-tune a COPY of `model` on `data`.
    cost   = integral of max(0, eval_loss - floor)   [censored at the budget]
    signed = integral of (eval_loss - floor)         [negative => positive transfer:
             the reformed model beats a purpose-built one of the same capacity]"""
    X, Yp, Yv = data
    m = copy.deepcopy(model); m.train()
    opt = torch.optim.Adam(m.parameters(), lr=FT_LR)
    n = len(X)
    cost, signed, step, reached = 0.0, 0.0, 0, -1
    while step < budget:
        perm = torch.randperm(n)
        for i in range(0, n, FT_BATCH):
            if step >= budget: break
            if step % EVAL_EVERY == 0:
                el = eval_loss(m, X, Yp)
                cost   += max(0.0, el-floor)*EVAL_EVERY
                signed += (el-floor)*EVAL_EVERY
                if reached < 0 and el <= floor*1.05: reached = step
            idx = perm[i:i+FT_BATCH]
            opt.zero_grad()
            o = m(X[idx])
            if kind == 'velocity':
                pos, spd = o
                loss = nn.functional.mse_loss(pos, Yp[idx]) + 0.1*nn.functional.mse_loss(spd, Yv[idx])
            else:
                loss = nn.functional.mse_loss(o, Yp[idx])
            loss.backward(); opt.step()
            step += 1
    m.eval()
    return m, cost, signed, (reached if reached >= 0 else budget), (reached < 0)


def run_seed(s):
    torch.set_num_threads(1)
    rng = np.random.default_rng(2000+s); torch.manual_seed(2000+s)

    print(f'  [seed {s}] zoo...', flush=True)
    models = [train_model(spec, rng) for spec in MODEL_SPECS]

    print(f'  [seed {s}] datasets + reference floors...', flush=True)
    data = {R: make_dataset(R, np.random.default_rng(7000+s+i)) for i, R in enumerate(REGIMES)}
    floors, refs = build_floors(s, data)
    for key, native in refs.items():                  # zoo models ARE their own references
        if native is not None:
            j = MODEL_NAMES.index(native)
            floors[key] = eval_loss(models[j], data[key[1]][0], data[key[1]][1])

    # ---- the map ----
    F, _ = schedule_stream(np.random.default_rng(6000+s))
    D = dmat_from_preds(np.stack([predict_stream(m, F) for m in models]))

    # ---- TEST 4: directed transition costs, capacity-fair floors ----
    print(f'  [seed {s}] transitions...', flush=True)
    rows, cost = [], {}
    for i, j in itertools.permutations(range(N_MODELS), 2):
        A, B = MODEL_NAMES[i], MODEL_NAMES[j]
        R = HOME[B]
        if R == HOME[A]:                              # same home regime -> no transition
            continue
        fl = floors[(ARCH[A], R)]                     # A's OWN architecture, on B's regime
        _, c, sg, tt, cens = finetune(models[i], KIND[A], data[R], fl)
        cost[(i, j)] = c
        rows.append(dict(seed=s, src=A, dst=B, target_regime=R,
                         d_beh=float(D[i, j]), d_cost=float(c), signed_cost=float(sg),
                         floor=float(fl), steps_to_reach=tt, censored=bool(cens),
                         positive_transfer=bool(sg < 0)))
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/transition_costs_seed{s}.csv', index=False)

    rho = spearmanr(df.d_beh, df.d_cost)[0]

    # symmetric-predictor ceiling: only over pairs measured in BOTH directions
    both = [(i, j) for i, j in itertools.combinations(range(N_MODELS), 2)
            if (i, j) in cost and (j, i) in cost]
    if both:
        sym  = [(cost[(i,j)]+cost[(j,i)])/2 for i, j in both]
        beh  = [D[i, j] for i, j in both]
        asym = [abs(cost[(i,j)]-cost[(j,i)])/max(cost[(i,j)], cost[(j,i)], 1e-9) for i, j in both]
        rho_sym = spearmanr(beh, sym)[0]
        med_asym = float(np.median(asym))
    else:
        rho_sym, med_asym = np.nan, np.nan

    # ---- TEST 5: geodesic detours ----
    print(f'  [seed {s}] chains...', flush=True)
    top = df.sort_values('d_cost', ascending=False).head(N_COSTLIEST)
    chains = []
    for _, r in top.iterrows():
        i = MODEL_NAMES.index(r.src)
        A, B = r.src, r.dst
        for k in range(N_MODELS):
            C = MODEL_NAMES[k]
            if C in (A, B) or HOME[C] in (HOME[A], HOME[B]): continue
            f1 = floors[(ARCH[A], HOME[C])]           # A's architecture throughout: the
            f2 = floors[(ARCH[A], HOME[B])]           # weights being moved are A's
            m1, c1, _, _, _ = finetune(models[i], KIND[A], data[HOME[C]], f1)
            _,  c2, _, _, _ = finetune(m1,        KIND[A], data[HOME[B]], f2)
            chains.append(dict(seed=s, src=A, via=C, dst=B,
                               direct=float(r.d_cost), leg1=float(c1), leg2=float(c2),
                               chained=float(c1+c2),
                               advantage=float((r.d_cost-(c1+c2))/max(r.d_cost, 1e-9)),
                               dominates=bool((c1+c2) < r.d_cost*(1-DOMINANCE_MARGIN))))
    dc = pd.DataFrame(chains)
    dc.to_csv(f'{OUT}/geodesic_chains_seed{s}.csv', index=False)

    print(f'  [seed {s}] done', flush=True)
    return dict(seed=s, rho=float(rho), rho_sym=float(rho_sym),
                censored_frac=float(df.censored.mean()),
                positive_transfer_frac=float(df.positive_transfer.mean()),
                asymmetry=med_asym,
                n_dominated_pairs=int(dc.groupby(['src','dst']).dominates.any().sum()) if len(dc) else 0,
                best_advantage=float(dc.advantage.max()) if len(dc) else 0.0)


def main():
    res = [run_seed(s) for s in range(N_SEEDS)]
    d = pd.DataFrame(res); d.to_csv(f'{OUT}/transition_cost_by_seed.csv', index=False)

    L = []
    def say(x): print(x); L.append(x)
    def iqr(x):
        x = np.asarray(x, float); x = x[~np.isnan(x)]
        if not len(x): return 'n/a'
        return f'{np.median(x):.3f} [{np.percentile(x,25):.3f}, {np.percentile(x,75):.3f}]'

    say('\n================ REGISTERED OUTCOMES (v2) ================\n')

    say('TEST 4 -- Does the geometry predict the cost of reform?')
    say(f'  Spearman rho(d_beh, d_cost), directed   : {iqr(d.rho)}   (bar {RHO_BAR})')
    say(f'  seeds with rho >= {RHO_FLOOR}                 : {int((d.rho >= RHO_FLOOR).sum())}/{N_SEEDS}')
    say(f'  rho(d_beh, SYMMETRIZED cost) -- ceiling  : {iqr(d.rho_sym)}')
    say(f'  median cost asymmetry |A->B vs B->A|    : {iqr(d.asymmetry)}')
    say(f'  transitions with POSITIVE transfer      : {iqr(d.positive_transfer_frac)}  (beat a fresh build)')
    say(f'  censored transitions                    : {iqr(d.censored_frac)}')
    ok4 = (np.median(d.rho) >= RHO_BAR) and int((d.rho >= RHO_FLOOR).sum()) >= PASS_BAR
    say(f'  H_predict -> {"PASS: behavioral distance predicts reform cost" if ok4 else "FAIL (null holds): THE GEOMETRY IS DECORATIVE"}')
    if not ok4:
        say('  >> Registered consequence: Paper XXIII has no engineering content as conceived,')
        say('  >> and should be re-centred or not written.  This outcome was registered before')
        say('  >> the run and is reportable as the result.')
    gap = float(np.median(d.rho_sym) - np.median(d.rho))
    say(f'  >> directed-vs-symmetric gap = {gap:.3f}.  A large gap means d_beh is not failing')
    say(f'  >> on its own terms but hitting a structural bound: a SYMMETRIC quantity cannot')
    say(f'  >> predict an ASYMMETRIC one, and cost is asymmetric.  Read with the asymmetry above.')

    say('\nTEST 5 -- Can a detour beat a direct move?')
    say(f'  seeds with >= 1 dominated pair : {int((d.n_dominated_pairs > 0).sum())}/{N_SEEDS} (bar {PASS_BAR})')
    say(f'  dominated pairs per seed       : {iqr(d.n_dominated_pairs)}')
    say(f'  best detour advantage          : {iqr(d.best_advantage)}  (margin {DOMINANCE_MARGIN})')
    ok5 = int((d.n_dominated_pairs > 0).sum()) >= PASS_BAR
    say(f'  H_geodesic -> {"PASS: a detour can be cheaper than a direct move" if ok5 else "FAIL (null holds): the direct transition is never dominated; geodesic reform is a metaphor"}')

    say('\nIf cost is strongly asymmetric while distance is symmetric, factorization space is')
    say('not a metric space, and the Riemannian framing was wrong from the start -- as')
    say('exploration 04 warned.  The governance reading of an asymmetric cost, if it')
    say('survives, is that reform is DIRECTIONAL: what an institution costs to leave is not')
    say('what it costs to return to.  That would be a finding, and it would be the paper.\n')

    open(f'{OUT}/xxiii_cost_summary.txt','w').write('\n'.join(L))
    print(f'Wrote {OUT}/xxiii_cost_summary.txt')

if __name__ == '__main__':
    main()
