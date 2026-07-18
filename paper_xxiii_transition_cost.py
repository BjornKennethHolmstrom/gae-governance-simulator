#!/usr/bin/env python3
"""
paper_xxiii_transition_cost.py
==============================
The load-bearing test.  Nothing in the repository establishes that the behavioral
geometry has any engineering content: in the Paper XIX architecture, switching is
FREE -- the adaptive controller selects an active model per step at zero cost.  The
geometry is a map with no travel on it.

This script puts travel on the map.  Moving from factorization A to factorization B
means retraining A's weights toward B's regime.  The cost of that move is the excess
error paid along the way.  Two preregistered tests follow.

  TEST 4  Does the geometry predict the cost of reform?
          d_beh(A,B)  = behavioral distance (prediction MSE difference), the map.
          d_cost(A->B)= cumulative excess error while fine-tuning A on B's home
                        regime, relative to B's own converged loss there.  Directed,
                        and not assumed symmetric.

          H_predict : Spearman rho(d_beh, d_cost) >= RHO_BAR, median across seeds,
                      and >= RHO_FLOOR in >= PASS_BAR seeds.
          NULL      : the geometry is DECORATIVE -- a pretty map that tells an
                      institution nothing about what a reform will cost.  If this
                      null holds, Paper XXIII should not be written.

  TEST 5  Can a detour beat a direct move?
          For the most expensive transitions, is there an intermediate C such that
          chaining A->C->B (weights carried through) costs less than A->B direct?

          H_geodesic : at least one dominated pair (chained cost < direct cost by
                       >= DOMINANCE_MARGIN) in >= PASS_BAR seeds.
          NULL       : the direct transition is never dominated, and "geodesic
                       reform" is a metaphor.

If H_geodesic passes, the series gains a design principle it does not have:
reforms fail because the PATH crosses an unstable intermediate, not because the
TARGET is wrong -- and a detour can be cheaper than a direct move.

Outputs (in ./xxiii_out/):
    transition_costs_seed{s}.csv     all 42 ordered pairs
    geodesic_chains_seed{s}.csv      chained vs direct, for the costliest pairs
    xxiii_cost_summary.txt

Runtime: ~45-70 min for N_SEEDS=10 on 8 cores, CPU-only.  Set N_SEEDS=2 to smoke-test.
"""

import os, itertools, copy
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from scipy.stats import spearmanr

from paper_xxiii_geometry_replication import (            # reuse, do not re-specify
    MODEL_SPECS, MODEL_NAMES, N_MODELS, INPUT_LEN, OFFSETS, MAX_OFF,
    rollout, schedule_stream, windows, make_model, train_model,
    predict_stream, dmat_from_preds, OUT,
)

# ======================================================================
# REGISTERED CONSTANTS -- fixed before any run
# ======================================================================
N_SEEDS          = 10
PASS_BAR         = 8      # of N_SEEDS
RHO_BAR          = 0.50   # Test 4: median Spearman across seeds
RHO_FLOOR        = 0.30   # Test 4: per-seed floor
FT_BUDGET        = 400    # fine-tuning optimizer steps per transition
FT_LR            = 5e-4
FT_BATCH         = 128
N_COSTLIEST      = 6      # Test 5: how many of the priciest direct pairs to probe
DOMINANCE_MARGIN = 0.10   # Test 5: chained must beat direct by >= 10%

torch.set_num_threads(os.cpu_count() or 8)

HOME = {name: spec[2] for name, spec in zip(MODEL_NAMES, MODEL_SPECS)}   # model -> home regime


def make_dataset(regime, rng, n_episodes=16, ep_len=600):
    Xs, Yps, Yvs = [], [], []
    for _ in range(n_episodes):
        F, P = rollout(regime, ep_len, rng)
        x, yp, yv = windows(F, P); Xs.append(x); Yps.append(yp); Yvs.append(yv)
    return torch.cat(Xs), torch.cat(Yps), torch.cat(Yvs)


@torch.no_grad()
def eval_loss(model, X, Yp, batch=512):
    model.eval(); tot, n = 0.0, 0
    for i in range(0, len(X), batch):
        o = model(X[i:i+batch])
        pos = o[0] if isinstance(o, tuple) else o
        tot += float(nn.functional.mse_loss(pos, Yp[i:i+batch], reduction='sum'))
        n += pos.numel()
    return tot / n


def finetune(model, kind, data, floor, budget=FT_BUDGET):
    """Fine-tune a COPY of `model` on `data`.  Cost = cumulative excess error above
    `floor` (the native model's converged loss there), integrated over the budget.
    Censored at the budget: a transition that never arrives still has a finite,
    comparable price, and the censoring is reported."""
    X, Yp, Yv = data
    m = copy.deepcopy(model); m.train()
    opt = torch.optim.Adam(m.parameters(), lr=FT_LR)
    n = len(X)
    excess, reached, step = 0.0, -1, 0
    while step < budget:
        perm = torch.randperm(n)
        for i in range(0, n, FT_BATCH):
            if step >= budget: break
            idx = perm[i:i+FT_BATCH]
            opt.zero_grad()
            o = m(X[idx])
            if kind == 'velocity':
                pos, spd = o
                loss = nn.functional.mse_loss(pos, Yp[idx]) + 0.1*nn.functional.mse_loss(spd, Yv[idx])
                track = float(nn.functional.mse_loss(pos, Yp[idx]))
            else:
                loss = nn.functional.mse_loss(o, Yp[idx]); track = float(loss)
            loss.backward(); opt.step()
            excess += max(0.0, track - floor)
            if reached < 0 and track <= floor * 1.05:
                reached = step
            step += 1
    m.eval()
    return m, excess, (reached if reached >= 0 else budget), (reached < 0)


def run_seed(s):
    rng = np.random.default_rng(2000 + s)
    torch.manual_seed(2000 + s)

    print(f'  training zoo...', flush=True)
    models = [train_model(spec, rng) for spec in MODEL_SPECS]
    kinds  = [spec[3] for spec in MODEL_SPECS]

    # ---- the map: behavioral distances on the shared regime-shift stream ----
    F, _ = schedule_stream(np.random.default_rng(6000 + s))
    D = dmat_from_preds(np.stack([predict_stream(m, F) for m in models]))

    # ---- per-regime datasets + each native model's floor on its own home ----
    regimes = sorted(set(HOME.values()))
    data = {r: make_dataset(r, np.random.default_rng(7000 + s + hash(r) % 100)) for r in regimes}
    floors = {}
    for j, name in enumerate(MODEL_NAMES):
        r = HOME[name]
        floors[name] = eval_loss(models[j], data[r][0], data[r][1])

    # ---- TEST 4: direct transition costs, all 42 ordered pairs ----
    rows, cost = [], {}
    for i, j in itertools.permutations(range(N_MODELS), 2):
        target_regime = HOME[MODEL_NAMES[j]]
        _, exc, tt, cens = finetune(models[i], kinds[i], data[target_regime], floors[MODEL_NAMES[j]])
        cost[(i, j)] = exc
        rows.append(dict(seed=s, src=MODEL_NAMES[i], dst=MODEL_NAMES[j],
                         target_regime=target_regime, d_beh=float(D[i, j]),
                         d_cost=float(exc), steps_to_reach=tt, censored=bool(cens)))
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/transition_costs_seed{s}.csv', index=False)

    rho, p = spearmanr(df.d_beh, df.d_cost)
    asym = float(np.mean([abs(cost[(i,j)] - cost[(j,i)]) / max(cost[(i,j)], cost[(j,i)], 1e-9)
                          for i, j in itertools.combinations(range(N_MODELS), 2)]))

    # ---- TEST 5: geodesic detours on the costliest direct pairs ----
    top = df.sort_values('d_cost', ascending=False).head(N_COSTLIEST)
    chains = []
    for _, r in top.iterrows():
        i = MODEL_NAMES.index(r.src); j = MODEL_NAMES.index(r.dst)
        direct = r.d_cost
        for k in range(N_MODELS):
            if k in (i, j): continue
            mid_regime = HOME[MODEL_NAMES[k]]
            m1, e1, _, _ = finetune(models[i], kinds[i], data[mid_regime], floors[MODEL_NAMES[k]])
            _,  e2, _, _ = finetune(m1, kinds[i], data[HOME[MODEL_NAMES[j]]], floors[MODEL_NAMES[j]])
            chains.append(dict(seed=s, src=MODEL_NAMES[i], via=MODEL_NAMES[k], dst=MODEL_NAMES[j],
                               direct=float(direct), leg1=float(e1), leg2=float(e2),
                               chained=float(e1+e2),
                               advantage=float((direct - (e1+e2)) / max(direct, 1e-9)),
                               dominates=bool((e1+e2) < direct * (1 - DOMINANCE_MARGIN))))
    dc = pd.DataFrame(chains)
    dc.to_csv(f'{OUT}/geodesic_chains_seed{s}.csv', index=False)

    return dict(seed=s, rho=float(rho), p=float(p),
                censored_frac=float(df.censored.mean()),
                asymmetry=asym,
                n_dominated_pairs=int(dc.groupby(['src','dst']).dominates.any().sum()),
                best_advantage=float(dc.advantage.max()) if len(dc) else 0.0)


def main():
    res = []
    for s in range(N_SEEDS):
        print(f'--- seed {s+1}/{N_SEEDS} ---', flush=True)
        res.append(run_seed(s))
    d = pd.DataFrame(res)
    d.to_csv(f'{OUT}/transition_cost_by_seed.csv', index=False)

    L = []
    def say(x): print(x); L.append(x)
    def iqr(x): return f'{np.median(x):.3f} [{np.percentile(x,25):.3f}, {np.percentile(x,75):.3f}]'

    say('\n================ REGISTERED OUTCOMES ================\n')

    say('TEST 4 -- Does the geometry predict the cost of reform?')
    say(f'  Spearman rho(d_beh, d_cost)   : {iqr(d.rho)}')
    say(f'  seeds with rho >= {RHO_FLOOR}      : {int((d.rho >= RHO_FLOOR).sum())}/{N_SEEDS}')
    say(f'  censored transitions          : {iqr(d.censored_frac)}  (never reached the floor in budget)')
    say(f'  cost asymmetry |A->B vs B->A| : {iqr(d.asymmetry)}  (distance is symmetric; cost need not be)')
    ok4 = (np.median(d.rho) >= RHO_BAR) and int((d.rho >= RHO_FLOOR).sum()) >= PASS_BAR
    say(f'  H_predict (median rho >= {RHO_BAR}, floor in >= {PASS_BAR}) -> '
        f'{"PASS: behavioral distance predicts reform cost" if ok4 else "FAIL (null holds): THE GEOMETRY IS DECORATIVE"}')
    if not ok4:
        say('  >> If this fails, Paper XXIII has no engineering content and should not be written.')
        say('  >> That outcome is registered, not a surprise, and is reportable as the result.')

    say('\nTEST 5 -- Can a detour beat a direct move?')
    say(f'  seeds with >= 1 dominated pair : {int((d.n_dominated_pairs > 0).sum())}/{N_SEEDS} (bar {PASS_BAR})')
    say(f'  dominated pairs per seed       : {iqr(d.n_dominated_pairs)}')
    say(f'  best detour advantage          : {iqr(d.best_advantage)}  (margin = {DOMINANCE_MARGIN})')
    ok5 = int((d.n_dominated_pairs > 0).sum()) >= PASS_BAR
    say(f'  H_geodesic -> '
        f'{"PASS: a detour can be cheaper than a direct move" if ok5 else "FAIL (null holds): the direct transition is never dominated; geodesic reform is a metaphor"}')

    say('\nNOTE: cost asymmetry is worth reading even if both tests fail.  Behavioral')
    say('distance is symmetric by construction; if transition cost is not, then the')
    say('object is not a metric space and the Riemannian framing was wrong from the')
    say('start -- which exploration 04 already warned about, and which would be a')
    say('finding rather than an embarrassment.\n')

    open(f'{OUT}/xxiii_cost_summary.txt', 'w').write('\n'.join(L))
    print(f'Wrote {OUT}/xxiii_cost_summary.txt')

if __name__ == '__main__':
    main()
