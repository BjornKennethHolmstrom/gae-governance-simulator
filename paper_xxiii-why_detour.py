#!/usr/bin/env python3
"""
paper_xxiii-why_detour.py
=========================
REGISTERED CONTROL.  Committed before the run.

The full run left one thing standing and unexplained: routing a reform through an
intermediate lowers its cost (H_geodesic, 10/10), and a badly chosen route is worse
than none (H_route, 10/10) -- but WHY a route helps is unresolved.  Two mechanism
stories already failed: H_waypoint (specific forms are hubs) was withdrawn because the
"hub" was trivially the cheapest regime to enter, and the entry-cost story came apart
under pooled analysis.

This control settles the one discriminating question left.

    GEODESIC hypothesis:  routing A -> C -> B helps because C lies BETWEEN A and B in
                          behavioral space -- leg 1 partially completes the move to B.
                          The benefit is specific to the (A, B) pair.

    CURRICULUM hypothesis: routing A -> C -> B helps because C is a good regime to
                          train on -- it improves the model regardless of destination.
                          The benefit is a property of C alone.  This is NOT geodesic,
                          and "route matters" would reduce to "some regimes are good
                          curricula."

DESIGN -- architecture held FIXED at h8-base, which removes the capacity confound that
affected every prior version (compressed_h2 etc.).  Sources, intermediates, and
destinations all range over the four regime-native h8-base models {normal, wind,
damped, blur}.  For every (source A, intermediate C, destination B) we measure the
onward-leg cost leg2(A, C, B): fine-tune A on C for one budget, then on B for one
budget, cost against the converged h8-base floor on B.  The null intermediate is
C = home(A).

REGISTERED DISCRIMINATORS
-------------------------
The onward-cost REDUCTION from routing through C rather than through home(A):

    Delta(A, C, B) = leg2(A, home_A, B) - leg2(A, C, B)

Q1  DESTINATION INDEPENDENCE (the decisive test).
    For each source A, rank intermediates C by Delta, separately for each destination
    B.  Take the rank correlation of these C-rankings BETWEEN destinations, and its
    median across destination-pairs and sources.
        >= WARMUP_TAU  -> the best intermediate is the same wherever you are going
                          => CURRICULUM.  The geodesic interpretation is WITHDRAWN.
        <= GEODESIC_TAU-> the best intermediate depends on the destination
                          => consistent with a genuine path.

Q2  BEHAVIORAL PATH.  Does the helpful C minimize the behavioral path length
    d_beh(A, C) + d_beh(C, B)?
        rho(Delta, -pathlength) >= PATH_RHO  -> C is literally "on the way"
                                                => geodesic.
        ~ 0                                   -> position of C is irrelevant
                                                => curriculum.

Q3  VARIANCE DECOMPOSITION.  Of the variance in Delta, how much is a main effect of C
    alone (curriculum) versus a source x destination x intermediate interaction
    (geodesic)?  Reported as partial eta-squared for the C-main-effect against the
    full interaction.

REGISTERED DECISION
-------------------
  CURRICULUM  if Q1 >= WARMUP_TAU  AND  Q2 rho < PATH_RHO.
              -> The surviving H_geodesic/H_route effects are real but are a
                 CURRICULUM effect, not a geodesic one.  XXIII reports "some regimes
                 are good intermediate training grounds," not "reform has geodesics."
                 This is a WEAKER and more honest claim, and it is registered as the
                 expected outcome given wind_h8 was chosen 46 times across unrelated
                 (A, B) pairs in the full run.

  GEODESIC    if Q1 <= GEODESIC_TAU  AND  Q2 rho >= PATH_RHO.
              -> Routing is genuinely path-dependent; the geodesic reading stands.

  MIXED / INCONCLUSIVE otherwise -> reported as such, no mechanism claimed.  The effect
              stays registered (it is real); the explanation does not (it is not shown).

Runtime: ~1.5-2.5 h for N_SEEDS=5 on 8 cores, CPU-only.  N_SEEDS=2 to smoke-test.
"""

import os, itertools, copy
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from scipy.stats import spearmanr

from paper_xxiii_geometry_replication_v2 import (
    rollout, schedule_stream, windows, train_arch, make_model,
    predict_stream, dmat_from_preds, OUT,
)

# ======================================================================
# REGISTERED CONSTANTS
# ======================================================================
N_SEEDS     = 5
REGIMES     = ['normal', 'wind', 'damped', 'blur']
HID, KIND   = 8, 'base'          # architecture held FIXED -- no capacity confound

FT_BUDGET   = 800
FT_LR       = 5e-4
FT_BATCH    = 128
EVAL_EVERY  = 20
REF_EPOCHS  = 20
REF_PATIENCE = 3

WARMUP_TAU   = 0.50     # Q1 >= this  -> destination-independent -> curriculum
GEODESIC_TAU = 0.20     # Q1 <= this  -> destination-dependent   -> geodesic
PATH_RHO     = 0.40     # Q2 >= this  -> C is on the behavioral path

torch.set_num_threads(os.cpu_count() or 8)


def make_dataset(regime, rng, n_episodes=16, ep_len=600):
    Xs, Yps, Yvs = [], [], []
    for _ in range(n_episodes):
        F, P = rollout(regime, ep_len, rng)
        x, yp, yv = windows(F, P); Xs.append(x); Yps.append(yp); Yvs.append(yv)
    return torch.cat(Xs), torch.cat(Yps), torch.cat(Yvs)


@torch.no_grad()
def eval_loss(model, X, Yp, batch=512):
    was = model.training; model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(X), batch):
        pos = model(X[i:i+batch])
        pos = pos[0] if isinstance(pos, tuple) else pos
        tot += float(nn.functional.mse_loss(pos, Yp[i:i+batch], reduction='sum').item())
        n += pos.numel()
    if was: model.train()
    return tot / n


def train_reference(data, val, rng):
    X, Yp, _ = data; Xv, Ypv, _ = val
    m = make_model(HID, KIND)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    best, best_state, bad = float('inf'), None, 0
    for _ in range(REF_EPOCHS):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), FT_BATCH):
            idx = perm[i:i+FT_BATCH]
            opt.zero_grad()
            loss = nn.functional.mse_loss(m(X[idx]), Yp[idx])
            loss.backward(); opt.step()
        vl = eval_loss(m, Xv, Ypv)
        if vl < best - 1e-6: best, best_state, bad = vl, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
            if bad >= REF_PATIENCE: break
    if best_state is not None: m.load_state_dict(best_state)
    m.eval(); return m, best


def finetune(model, data, floor, budget=FT_BUDGET):
    X, Yp, _ = data
    m = copy.deepcopy(model); m.train()
    opt = torch.optim.Adam(m.parameters(), lr=FT_LR)
    cost, step = 0.0, 0
    while step < budget:
        perm = torch.randperm(len(X))
        for i in range(0, len(X), FT_BATCH):
            if step >= budget: break
            if step % EVAL_EVERY == 0:
                cost += max(0.0, eval_loss(m, X, Yp) - floor) * EVAL_EVERY
            idx = perm[i:i+FT_BATCH]
            opt.zero_grad()
            loss = nn.functional.mse_loss(m(X[idx]), Yp[idx])
            loss.backward(); opt.step(); step += 1
    m.eval(); return m, cost


def run_seed(s):
    torch.set_num_threads(1)
    rng = np.random.default_rng(3000 + s); torch.manual_seed(3000 + s)

    data = {R: make_dataset(R, np.random.default_rng(3100 + s + i)) for i, R in enumerate(REGIMES)}
    val  = {R: make_dataset(R, np.random.default_rng(3200 + s + i), n_episodes=6) for i, R in enumerate(REGIMES)}

    print(f'  [seed {s}] native models + floors...', flush=True)
    native, floor = {}, {}
    for R in REGIMES:
        m, fl = train_reference(data[R], val[R], rng)   # a converged h8-base on R
        native[R] = m; floor[R] = fl                    # IS the reference floor for R

    # behavioral distances between the native models, on the shared regime-shift stream
    F, _ = schedule_stream(np.random.default_rng(3300 + s))
    order = REGIMES
    preds = np.stack([predict_stream(native[R], F) for R in order])
    Db = dmat_from_preds(preds)
    dbeh = {(order[i], order[j]): float(Db[i, j])
            for i in range(len(order)) for j in range(len(order))}

    print(f'  [seed {s}] source x intermediate x destination cube...', flush=True)
    # leg1[(A,C)]: native model of regime A, fine-tuned on regime C
    leg1_model = {}
    for A in REGIMES:
        for C in REGIMES:
            m, _ = finetune(native[A], data[C], floor[C])
            leg1_model[(A, C)] = m

    rows = []
    for A in REGIMES:
        for C in REGIMES:
            for B in REGIMES:
                if B == A:  # destination == source's home is not a reform
                    continue
                _, leg2 = finetune(leg1_model[(A, C)], data[B], floor[B])
                rows.append(dict(seed=s, source=A, intermediate=C, dest=B,
                                 leg2=float(leg2), is_null=(C == A),
                                 pathlen=dbeh[(A, C)] + dbeh[(C, B)]))
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/why_detour_seed{s}.csv', index=False)
    return df


def analyse(all_df, say):
    def iqr(x):
        x = np.asarray(x, float); x = x[~np.isnan(x)]
        return 'n/a' if not len(x) else f'{np.median(x):.3f} [{np.percentile(x,25):.3f}, {np.percentile(x,75):.3f}]'

    # Delta(A,C,B) = leg2 via home(A)  -  leg2 via C
    rows = []
    for (s, A, B), g in all_df.groupby(['seed', 'source', 'dest']):
        null = g[g.is_null]
        if not len(null): continue
        base = float(null.leg2.iloc[0])
        for r in g[~g.is_null].itertuples():
            rows.append(dict(seed=s, source=A, dest=B, intermediate=r.intermediate,
                             delta=base - r.leg2, pathlen=r.pathlen, leg2=r.leg2))
    d = pd.DataFrame(rows)

    say('\n=============== WHY DOES THE DETOUR HELP? ===============\n')

    # ---- Q1: destination independence ----
    # For each (seed, source), rank intermediates by delta per destination; correlate
    # those rankings between destinations.  High => same best C wherever you go.
    q1 = []
    for (s, A), g in d.groupby(['seed', 'source']):
        piv = g.pivot_table(index='intermediate', columns='dest', values='delta')
        dests = list(piv.columns)
        for b1, b2 in itertools.combinations(dests, 2):
            v = piv[[b1, b2]].dropna()
            if len(v) >= 3:
                q1.append(spearmanr(v[b1], v[b2])[0])
    q1 = [x for x in q1 if not np.isnan(x)]
    say('Q1 -- is the best intermediate the SAME regardless of destination?')
    say(f'  cross-destination rank correlation of C-helpfulness : {iqr(q1)}')
    say(f'     >= {WARMUP_TAU} => curriculum (destination-independent)')
    say(f'     <= {GEODESIC_TAU} => geodesic (destination-dependent)')
    q1med = float(np.median(q1)) if q1 else np.nan

    # ---- Q2: behavioral path ----
    # Does a larger delta go with a SHORTER behavioral path A->C->B?
    q2 = []
    for (s, A, B), g in d.groupby(['seed', 'source', 'dest']):
        if len(g) >= 3:
            q2.append(spearmanr(g.delta, -g.pathlen)[0])
    q2 = [x for x in q2 if not np.isnan(x)]
    say('\nQ2 -- does the helpful intermediate lie on the behavioral PATH A->C->B?')
    say(f'  rho(delta, -pathlength) : {iqr(q2)}   (>= {PATH_RHO} => C is on the way)')
    q2med = float(np.median(q2)) if q2 else np.nan

    # ---- Q3: variance decomposition ----
    # main effect of C alone vs the full (source,dest,intermediate) cell
    gm = d.delta.mean()
    ss_tot = float(((d.delta - gm) ** 2).sum())
    ss_C = float(sum(len(g) * (g.delta.mean() - gm) ** 2 for _, g in d.groupby('intermediate')))
    ss_cell = float(sum(len(g) * (g.delta.mean() - gm) ** 2
                        for _, g in d.groupby(['source', 'dest', 'intermediate'])))
    say('\nQ3 -- variance in the benefit: intermediate ALONE vs the full triple')
    say(f'  variance explained by intermediate C alone      : {ss_C/ss_tot:6.1%}   [curriculum]')
    say(f'  variance explained by (source,dest,intermediate): {ss_cell/ss_tot:6.1%}   [ceiling]')
    say(f'  interaction beyond the C main effect            : {(ss_cell-ss_C)/ss_tot:6.1%}   [geodesic]')

    # which intermediate is best, and how often, per destination
    say('\n  best intermediate by destination (if constant across rows => curriculum):')
    for B in REGIMES:
        sub = d[d.dest == B]
        if not len(sub): continue
        best = sub.groupby('intermediate').delta.mean().idxmax()
        say(f'    -> {B:8}: {best}')

    # ---- registered decision ----
    say('\n--- REGISTERED DECISION ---')
    curriculum = (q1med >= WARMUP_TAU) and (q2med < PATH_RHO)
    geodesic   = (q1med <= GEODESIC_TAU) and (q2med >= PATH_RHO)
    if curriculum:
        say('  CURRICULUM.  The detour benefit is destination-independent and does not')
        say('  track behavioral path length.  H_geodesic and H_route are REAL effects but')
        say('  are a CURRICULUM effect: some regimes are good intermediate training grounds,')
        say('  regardless of where the reform is headed.  The geodesic INTERPRETATION is')
        say('  withdrawn.  XXIII claims a curriculum effect, not reform geodesics.')
    elif geodesic:
        say('  GEODESIC.  The benefit depends on the destination and tracks behavioral path')
        say('  length.  Routing is genuinely path-dependent; the geodesic reading stands.')
    else:
        say('  MIXED / INCONCLUSIVE.  The effect is real and stays registered; the mechanism')
        say('  is not shown and no mechanism is claimed.  XXIII reports the effect and states')
        say('  plainly that why a route helps is unresolved.')
    say('')


def main():
    import sys
    stage_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else N_SEEDS
    dfs = [run_seed(s) for s in range(stage_seeds)]
    all_df = pd.concat(dfs)
    L = []
    def say(x): print(x); L.append(x)
    analyse(all_df, say)
    open(f'{OUT}/why_detour_summary.txt', 'w').write('\n'.join(L))
    print(f'Wrote {OUT}/why_detour_summary.txt')

if __name__ == '__main__':
    main()
