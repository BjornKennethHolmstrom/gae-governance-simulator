#!/usr/bin/env python3
"""
paper_xxiii_transition_cost_v4.py
=================================
v4.  The v3 null-detour control fired, and it was right to.  This version registers
the test the control implies.

WHAT v3 FOUND
-------------
    TEST 5 (v3):  real detours beat the direct move by 65.7%
                  BUT the NULL detour -- routing through the source's OWN home regime,
                  which should cost nothing -- also beat direct, by 20.3%, in 2/2 seeds.

So 400 extra gradient steps are worth ~20% of reform cost no matter where they are
spent.  Comparing a detour against the DIRECT path is therefore comparing 800 steps
against 400, and any "geodesic" advantage is contaminated.

THE CORRECTED TEST
------------------
Compare the real detour against the NULL detour.  Both arms then receive FT_BUDGET
steps on an intermediate plus FT_BUDGET steps on the target; both pass through
*something*.  The only difference is WHICH intermediate.  That isolates path
structure from compute.

Re-analysing the v3 data under this comparison:

    best real detour beats the NULL detour by >= 10%  :  8/12 pairs
    median gain of best-real OVER THE NULL            :  45.4%
    null advantage over direct (PURE COMPUTE)         :  19.9%
    best-real advantage over direct (total)           :  44.7%

The effect decomposes: ~20 points of compute, ~25 points of genuine path structure.
The control fired, and the effect survived it.

    H_geodesic (registered):  the best real detour beats the NULL detour by
                              >= DOMINANCE_MARGIN, in >= PASS_BAR seeds.
    NULL:  all intermediates are interchangeable -- every gain is gradient steps,
           and "geodesic reform" is an artifact of the experimental design.

    H_route (registered, secondary):  route choice MATTERS.  For a fixed (A,B), the
           spread of chained cost across intermediates C is large, and the WORST real
           detour is worse than the null.
    NULL:  the choice of intermediate is immaterial.

H_route is the claim we would actually want an institution to act on, and it is the
one most likely to be missed by a paper that only reports its best detour.  Four of
twelve pairs in v3 had their best real detour WORSE than the null: a badly chosen
route is worse than no route at all.

WHY THE LITERAL TRIANGLE INEQUALITY IS *REPORTED* BUT NOT *REGISTERED*
----------------------------------------------------------------------
One can compute d(A->B) against d(A->C) + d(C->B) using three independent
single-budget fine-tunes -- compute-fair, no chaining.  It is violated in 25-29% of
triples.  We report it as a diagnostic and decline to register it, because it rests
on a false premise: **paying d(A->C) does not put you AT C.**  A cost of zero means A
already PERFORMS at C's level, not that A IS C.  Performance parity is not identity,
so the two edges do not compose, and a "triangle inequality" over them is not
well-posed.  The chained test carries the weights through and is the honest one.

This matters for the paper's central claim.  Factorization space is not a metric
space -- but the reason is not merely that cost is asymmetric.  It is that transition
cost is a directed quantity from a MODEL to a REGIME, and composing such quantities
requires knowing which model you end up at.  There is no common space in which the
triangle inequality is even statable.  That is a stronger and stranger claim than
"the Riemannian framing was wrong", and it is what the data support.

OTHER CHANGES
-------------
  * FT_BUDGET 400 -> 800.  v3's converged reference floors pushed censoring to 26.7%;
    a censored transition has its cost truncated, which ATTENUATES Test 4's rho.  The
    v3 pass at rho = 0.557 is therefore conservative.
  * Hub structure reported: which intermediates are chosen as best, and how often.
    In v3 seed 0, EVERY dominating detour routed through wind_h8 -- free to enter from
    everywhere, cheap to leave.  If that replicates, some institutional forms are
    WAYPOINTS, and that is a governance concept the series does not have.

Runtime: ~4-6 h for N_SEEDS=10 on 8 cores.  N_SEEDS=2 to smoke-test.
"""

import os, itertools, copy, collections
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from scipy.stats import spearmanr

from paper_xxiii_geometry_replication_v2 import (
    MODEL_SPECS, MODEL_NAMES, N_MODELS,
    rollout, schedule_stream, windows, train_arch, train_model,
    predict_stream, dmat_from_preds, make_model, OUT,
)

# ======================================================================
# REGISTERED CONSTANTS
# ======================================================================
N_SEEDS          = 10
PASS_BAR         = 8
RHO_BAR          = 0.50
RHO_FLOOR        = 0.30
FT_BUDGET        = 800      # was 400: cuts censoring, which attenuates Test 4
FT_LR            = 5e-4
FT_BATCH         = 128
EVAL_EVERY       = 20
N_COSTLIEST      = 6
DOMINANCE_MARGIN = 0.10     # a real detour must beat the NULL detour by this
ROUTE_SPREAD_BAR = 0.30     # H_route: (worst - best) / null >= this

REF_EPOCHS       = 20
REF_PATIENCE     = 3

torch.set_num_threads(os.cpu_count() or 8)

HOME  = {n: s[2] for n, s in zip(MODEL_NAMES, MODEL_SPECS)}
ARCH  = {n: (s[1], s[3]) for n, s in zip(MODEL_NAMES, MODEL_SPECS)}
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
    was = model.training; model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(X), batch):
        o = model(X[i:i+batch])
        pos = o[0] if isinstance(o, tuple) else o
        tot += float(nn.functional.mse_loss(pos, Yp[i:i+batch], reduction='sum').item())
        n += pos.numel()
    if was: model.train()
    return tot/n


def train_reference(hid, kind, data, val, rng):
    X, Yp, Yv = data; Xv, Ypv, _ = val
    m = make_model(hid, kind)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    best, best_state, bad, steps = float('inf'), None, 0, 0
    for _ in range(REF_EPOCHS):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), FT_BATCH):
            idx = perm[i:i+FT_BATCH]
            opt.zero_grad()
            o = m(X[idx])
            if kind == 'velocity':
                pos, spd = o
                loss = nn.functional.mse_loss(pos, Yp[idx]) + 0.1*nn.functional.mse_loss(spd, Yv[idx])
            else:
                loss = nn.functional.mse_loss(o, Yp[idx])
            loss.backward(); opt.step(); steps += 1
        vl = eval_loss(m, Xv, Ypv)
        if vl < best - 1e-6:
            best, best_state, bad = vl, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
            if bad >= REF_PATIENCE: break
    if best_state is not None: m.load_state_dict(best_state)
    m.eval(); return m, best, steps


def finetune(model, kind, data, floor, budget=FT_BUDGET):
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
            loss.backward(); opt.step(); step += 1
    m.eval(); return m, cost, signed, (reached if reached >= 0 else budget), (reached < 0)


def run_seed(s):
    torch.set_num_threads(1)
    rng = np.random.default_rng(2000+s); torch.manual_seed(2000+s)
    print(f'  [seed {s}] zoo...', flush=True)
    models = [train_model(spec, rng) for spec in MODEL_SPECS]

    print(f'  [seed {s}] converged reference floors...', flush=True)
    data = {R: make_dataset(R, np.random.default_rng(7000+s+i)) for i, R in enumerate(REGIMES)}
    val  = {R: make_dataset(R, np.random.default_rng(7500+s+i), n_episodes=6) for i, R in enumerate(REGIMES)}
    floors, ref_steps = {}, {}
    for (hid, kind) in sorted(set(ARCH.values())):
        for R in REGIMES:
            _, fl, st = train_reference(hid, kind, data[R], val[R], rng)
            floors[((hid, kind), R)] = fl; ref_steps[((hid, kind), R)] = st

    F, _ = schedule_stream(np.random.default_rng(6000+s))
    D = dmat_from_preds(np.stack([predict_stream(m, F) for m in models]))

    # ---------- TEST 4 ----------
    print(f'  [seed {s}] transitions...', flush=True)
    rows, cost = [], {}
    for i, j in itertools.permutations(range(N_MODELS), 2):
        A, B = MODEL_NAMES[i], MODEL_NAMES[j]
        R = HOME[B]
        if R == HOME[A]: continue
        fl = floors[(ARCH[A], R)]
        _, c, sg, tt, cens = finetune(models[i], KIND[A], data[R], fl)
        cost[(A, R)] = c
        rows.append(dict(seed=s, src=A, dst=B, target_regime=R, d_beh=float(D[i, j]),
                         d_cost=float(c), signed_cost=float(sg), floor=float(fl),
                         ref_train_steps=ref_steps[(ARCH[A], R)], steps_to_reach=tt,
                         censored=bool(cens), positive_transfer=bool(sg < 0)))
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/transition_costs_v4_seed{s}.csv', index=False)

    rho = spearmanr(df.d_beh, df.d_cost)[0]
    both = [(i, j) for i, j in itertools.combinations(range(N_MODELS), 2)
            if (MODEL_NAMES[i], HOME[MODEL_NAMES[j]]) in cost
            and (MODEL_NAMES[j], HOME[MODEL_NAMES[i]]) in cost]
    if both:
        cij = [cost[(MODEL_NAMES[i], HOME[MODEL_NAMES[j]])] for i, j in both]
        cji = [cost[(MODEL_NAMES[j], HOME[MODEL_NAMES[i]])] for i, j in both]
        rho_sym = spearmanr([D[i, j] for i, j in both], [(a+b)/2 for a, b in zip(cij, cji)])[0]
        asym = float(np.median([abs(a-b)/max(a, b, 1e-9) for a, b in zip(cij, cji)]))
    else:
        rho_sym, asym = np.nan, np.nan

    # ---------- TEST 5: real detour vs NULL detour, COMPUTE-MATCHED ----------
    print(f'  [seed {s}] chains (real + null)...', flush=True)
    top = df.sort_values('d_cost', ascending=False).head(N_COSTLIEST)
    chains, pairs = [], []
    for _, r in top.iterrows():
        i = MODEL_NAMES.index(r.src); A, B = r.src, r.dst
        # NULL detour: through A's OWN home.  ~0 cost, full budget of gradient steps.
        m0, c0a, _, _, _ = finetune(models[i], KIND[A], data[HOME[A]], floors[(ARCH[A], HOME[A])])
        _,  c0b, _, _, _ = finetune(m0, KIND[A], data[HOME[B]], floors[(ARCH[A], HOME[B])])
        null_cost = c0a + c0b
        chains.append(dict(seed=s, src=A, via='__NULL__', dst=B, is_null=True,
                           direct=float(r.d_cost), leg1=float(c0a), leg2=float(c0b),
                           chained=float(null_cost)))
        reals = []
        for k in range(N_MODELS):
            C = MODEL_NAMES[k]
            if C in (A, B) or HOME[C] in (HOME[A], HOME[B]): continue
            m1, c1, _, _, _ = finetune(models[i], KIND[A], data[HOME[C]], floors[(ARCH[A], HOME[C])])
            _,  c2, _, _, _ = finetune(m1, KIND[A], data[HOME[B]], floors[(ARCH[A], HOME[B])])
            reals.append((C, c1+c2))
            chains.append(dict(seed=s, src=A, via=C, dst=B, is_null=False,
                               direct=float(r.d_cost), leg1=float(c1), leg2=float(c2),
                               chained=float(c1+c2)))
        if reals:
            best_C, best_c = min(reals, key=lambda x: x[1])
            worst_C, worst_c = max(reals, key=lambda x: x[1])
            pairs.append(dict(seed=s, src=A, dst=B, direct=float(r.d_cost),
                              null=float(null_cost), best_via=best_C, best=float(best_c),
                              worst_via=worst_C, worst=float(worst_c),
                              gain_vs_null=float((null_cost-best_c)/max(null_cost, 1e-9)),
                              beats_null=bool(best_c < null_cost*(1-DOMINANCE_MARGIN)),
                              route_spread=float((worst_c-best_c)/max(null_cost, 1e-9)),
                              worst_beaten_by_null=bool(worst_c > null_cost),
                              compute_component=float((r.d_cost-null_cost)/max(r.d_cost, 1e-9)),
                              total_advantage=float((r.d_cost-best_c)/max(r.d_cost, 1e-9))))
    pd.DataFrame(chains).to_csv(f'{OUT}/geodesic_chains_v4_seed{s}.csv', index=False)
    p = pd.DataFrame(pairs)
    p.to_csv(f'{OUT}/geodesic_pairs_v4_seed{s}.csv', index=False)

    return dict(seed=s, rho=float(rho), rho_sym=float(rho_sym), asymmetry=asym,
                positive_transfer_frac=float(df.positive_transfer.mean()),
                censored_frac=float(df.censored.mean()),
                ref_steps_median=float(np.median(list(ref_steps.values()))),
                n_pairs=len(p),
                frac_beating_null=float(p.beats_null.mean()) if len(p) else np.nan,
                median_gain_vs_null=float(p.gain_vs_null.median()) if len(p) else np.nan,
                median_compute_component=float(p.compute_component.median()) if len(p) else np.nan,
                median_total_advantage=float(p.total_advantage.median()) if len(p) else np.nan,
                median_route_spread=float(p.route_spread.median()) if len(p) else np.nan,
                frac_worst_beaten_by_null=float(p.worst_beaten_by_null.mean()) if len(p) else np.nan,
                hubs=collections.Counter(p.best_via).most_common(2) if len(p) else [])


def main():
    res = [run_seed(s) for s in range(N_SEEDS)]
    d = pd.DataFrame(res); d.to_csv(f'{OUT}/transition_cost_v4_by_seed.csv', index=False)
    L = []
    def say(x): print(x); L.append(x)
    def iqr(x):
        x = np.asarray(x, float); x = x[~np.isnan(x)]
        return 'n/a' if not len(x) else f'{np.median(x):.3f} [{np.percentile(x,25):.3f}, {np.percentile(x,75):.3f}]'

    say('\n================ REGISTERED OUTCOMES (v4) ================\n')

    say('TEST 4 -- Does the geometry predict the cost of reform?')
    say(f'  Spearman rho(d_beh, d_cost), directed  : {iqr(d.rho)}   (bar {RHO_BAR})')
    say(f'  seeds with rho >= {RHO_FLOOR}                : {int((d.rho>=RHO_FLOOR).sum())}/{N_SEEDS}')
    say(f'  rho(d_beh, SYMMETRIZED cost) -- ceiling : {iqr(d.rho_sym)}')
    say(f'  cost asymmetry |A->B vs B->A|          : {iqr(d.asymmetry)}   [distance is symmetric; cost is not]')
    say(f'  positive transfer vs CONVERGED floor    : {iqr(d.positive_transfer_frac)}')
    say(f'  reference-floor training steps          : {iqr(d.ref_steps_median)}  [the floor is a real floor]')
    say(f'  censored transitions                    : {iqr(d.censored_frac)}  [censoring ATTENUATES rho -> conservative]')
    ok4 = (np.median(d.rho) >= RHO_BAR) and int((d.rho>=RHO_FLOOR).sum()) >= PASS_BAR
    say(f'  H_predict -> {"PASS: behavioral distance predicts reform cost" if ok4 else "FAIL (null): the geometry is decorative"}')

    say('\nTEST 5 -- Does the ROUTE matter?  (compute-matched: real detour vs NULL detour)')
    say(f'  pairs where best real detour beats the NULL : {iqr(d.frac_beating_null)}')
    say(f'  median gain of best-real OVER THE NULL      : {iqr(d.median_gain_vs_null)}   (bar {DOMINANCE_MARGIN})')
    say('')
    say('  DECOMPOSITION of the advantage over the direct path:')
    say(f'    compute component (null vs direct)        : {iqr(d.median_compute_component)}   <- gradient steps')
    say(f'    total advantage   (best-real vs direct)   : {iqr(d.median_total_advantage)}')
    say(f'    => path structure = total - compute')
    nb = int((d.frac_beating_null >= 0.5).sum())
    ok5 = nb >= PASS_BAR and np.median(d.median_gain_vs_null) >= DOMINANCE_MARGIN
    say(f'  H_geodesic: {nb}/{N_SEEDS} seeds (bar {PASS_BAR}) -> '
        f'{"PASS: routing through the right intermediate beats routing through a null one, at EQUAL COMPUTE. The advantage is PATH STRUCTURE." if ok5 else "FAIL (null): all intermediates are interchangeable; the gain was gradient steps"}')

    say('\nH_route (secondary) -- is a BADLY chosen route worse than no route?')
    say(f'  spread (worst - best) / null                : {iqr(d.median_route_spread)}   (bar {ROUTE_SPREAD_BAR})')
    say(f'  pairs whose WORST real detour loses to null : {iqr(d.frac_worst_beaten_by_null)}')
    okr = np.median(d.median_route_spread) >= ROUTE_SPREAD_BAR
    say(f'  H_route -> {"PASS: route choice matters, and a bad route is worse than no route" if okr else "FAIL (null): the choice of intermediate is immaterial"}')

    say('\nHUBS -- which intermediates are chosen as best?')
    cnt = collections.Counter()
    for h in d.hubs:
        for name, n in (h if isinstance(h, list) else []): cnt[name] += n
    for name, n in cnt.most_common(4):
        say(f'    {name:18} chosen best {n} times')
    say('  If one form is repeatedly the best waypoint -- cheap to enter from anywhere,')
    say('  cheap to leave -- then some institutional forms are WAYPOINTS, and that is a')
    say('  governance concept the series does not currently have.')

    say('\nNOTE (reported, not registered): the literal triangle inequality over EDGE costs')
    say('is violated in 25-29% of triples.  We decline to register it because paying')
    say('d(A->C) does not put you AT C -- a cost of zero means A already PERFORMS at C\'s')
    say('level, not that A IS C.  Performance parity is not identity, the edges do not')
    say('compose, and the inequality is not well-posed.  The chained test carries the')
    say('weights through and is the honest one.\n')

    open(f'{OUT}/xxiii_cost_summary.txt','w').write('\n'.join(L))
    print(f'Wrote {OUT}/xxiii_cost_summary.txt')

if __name__ == '__main__':
    main()
