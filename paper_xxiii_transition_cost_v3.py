#!/usr/bin/env python3
"""
paper_xxiii_transition_cost_v3.py
=================================
v3.  Two fixes.  The v2 results are promising enough that they must be attacked
harder, not celebrated.

v2 RESULT (3-zoo / 2-seed smoke):
    TEST 4  rho(d_beh, d_cost) = 0.655  (bar 0.50)  -- PASS
            symmetric-predictor ceiling = 0.697; gap 0.042
            cost asymmetry = 0.685; positive transfer in 58% of transitions
    TEST 5  every seed had dominating detours; best advantage 0.64  -- PASS

FIX 1 -- THE COMPUTE CONFOUND IN TEST 5 (the important one)
-----------------------------------------------------------
The chained path A->C->B receives FT_BUDGET optimizer steps on leg 1 and another
FT_BUDGET on leg 2.  The direct path A->B receives FT_BUDGET.  So the chain gets
twice the gradient steps, and some dominating chains had leg1 == 0.000 exactly --
a free detour that nonetheless moved the weights for 400 steps.  Whether a detour
is PATH STRUCTURE or merely EXTRA TRAINING is not distinguishable in v2.

The control that settles it is the NULL DETOUR: route A through its OWN home regime
before going to B.  Leg 1 then costs ~0 (A is already converged there) while still
consuming a full FT_BUDGET of gradient steps.

    H_geodesic (revised):  a real detour beats the direct move by >= DOMINANCE_MARGIN
                           AND the null detour does NOT (it ties direct, within
                           NULL_TIE_TOL).
    NULL:  the null detour beats direct too -- in which case the advantage is bought
           with gradient steps, not with geometry, and "geodesic reform" is an
           artifact of the experimental design rather than a property of the space.

This is the test we would want run against us, so we run it ourselves.

FIX 2 -- THE FLOOR MUST BE A REAL FLOOR
---------------------------------------
"Positive transfer" means the fine-tuned model beats a PURPOSE-BUILT model of the
same architecture on the target regime.  In v2 the purpose-built reference was
trained for ~660 gradient steps while fine-tuning got 400 -- so a weak reference
would inflate the transfer rate.  v3 trains reference floors to convergence
(REF_EPOCHS, with early stopping on a held-out split) and reports the reference's
own training budget alongside the result, so the claim can be audited.

ALSO REPORTED (new, and it is the real object)
----------------------------------------------
The TRIANGLE INEQUALITY VIOLATION RATE: the fraction of ordered triples (A,C,B)
with d_cost(A->C) + d_cost(C->B) < d_cost(A->B).  In a metric space this is zero by
definition.  Behavioral distance is symmetric and satisfies the triangle inequality;
transition cost, on this evidence, does neither.  If that survives the null-detour
control, then factorization space is not a metric space -- it is a DIRECTED COST
STRUCTURE -- the Riemannian framing of exploration 04 was wrong from the start, and
the governance reading is that reform is DIRECTIONAL and PATH-DEPENDENT: what an
institution costs to leave is not what it costs to return to, and the route matters.

Runtime: ~3-4 h for N_SEEDS=10 on 8 cores.  N_SEEDS=2 to smoke-test.
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
    predict_stream, dmat_from_preds, make_model, OUT,
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
EVAL_EVERY       = 20
N_COSTLIEST      = 6
DOMINANCE_MARGIN = 0.10
NULL_TIE_TOL     = 0.10     # the null detour must land within 10% of direct

REF_EPOCHS       = 20       # reference floors trained to convergence, not to a budget
REF_PATIENCE     = 3        # early stopping on a held-out split

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
    """A REAL floor: trained to convergence with early stopping, not to a fixed budget.
    If the purpose-built institution is under-trained, every reform 'beats' it."""
    X, Yp, Yv = data
    Xv, Ypv, _ = val
    m = make_model(hid, kind)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    best, best_state, bad, steps = float('inf'), None, 0, 0
    for ep in range(REF_EPOCHS):
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
    m.eval()
    return m, best, steps


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
            loss.backward(); opt.step()
            step += 1
    m.eval()
    return m, cost, signed, (reached if reached >= 0 else budget), (reached < 0)


def run_seed(s):
    torch.set_num_threads(1)
    rng = np.random.default_rng(2000+s); torch.manual_seed(2000+s)

    print(f'  [seed {s}] zoo...', flush=True)
    models = [train_model(spec, rng) for spec in MODEL_SPECS]

    print(f'  [seed {s}] datasets + CONVERGED reference floors...', flush=True)
    data = {R: make_dataset(R, np.random.default_rng(7000+s+i)) for i, R in enumerate(REGIMES)}
    val  = {R: make_dataset(R, np.random.default_rng(7500+s+i), n_episodes=6) for i, R in enumerate(REGIMES)}

    floors, ref_steps = {}, {}
    for (hid, kind) in sorted(set(ARCH.values())):
        for R in REGIMES:
            _, fl, st = train_reference(hid, kind, data[R], val[R], rng)
            floors[((hid, kind), R)] = fl
            ref_steps[((hid, kind), R)] = st

    F, _ = schedule_stream(np.random.default_rng(6000+s))
    D = dmat_from_preds(np.stack([predict_stream(m, F) for m in models]))

    # ---------- TEST 4: directed transition costs ----------
    print(f'  [seed {s}] transitions...', flush=True)
    rows, cost = [], {}
    for i, j in itertools.permutations(range(N_MODELS), 2):
        A, B = MODEL_NAMES[i], MODEL_NAMES[j]
        R = HOME[B]
        if R == HOME[A]: continue
        fl = floors[(ARCH[A], R)]
        _, c, sg, tt, cens = finetune(models[i], KIND[A], data[R], fl)
        cost[(A, R)] = c
        rows.append(dict(seed=s, src=A, dst=B, target_regime=R,
                         d_beh=float(D[i, j]), d_cost=float(c), signed_cost=float(sg),
                         floor=float(fl), ref_train_steps=ref_steps[(ARCH[A], R)],
                         steps_to_reach=tt, censored=bool(cens),
                         positive_transfer=bool(sg < 0)))
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/transition_costs_v3_seed{s}.csv', index=False)

    rho = spearmanr(df.d_beh, df.d_cost)[0]
    both = [(i, j) for i, j in itertools.combinations(range(N_MODELS), 2)
            if (MODEL_NAMES[i], HOME[MODEL_NAMES[j]]) in cost
            and (MODEL_NAMES[j], HOME[MODEL_NAMES[i]]) in cost]
    if both:
        cij = [cost[(MODEL_NAMES[i], HOME[MODEL_NAMES[j]])] for i, j in both]
        cji = [cost[(MODEL_NAMES[j], HOME[MODEL_NAMES[i]])] for i, j in both]
        sym = [(a+b)/2 for a, b in zip(cij, cji)]
        beh = [D[i, j] for i, j in both]
        rho_sym = spearmanr(beh, sym)[0]
        asym = float(np.median([abs(a-b)/max(a, b, 1e-9) for a, b in zip(cij, cji)]))
    else:
        rho_sym, asym = np.nan, np.nan

    # ---------- TEST 5: detours, WITH the null-detour control ----------
    print(f'  [seed {s}] chains + null controls...', flush=True)
    top = df.sort_values('d_cost', ascending=False).head(N_COSTLIEST)
    chains = []
    for _, r in top.iterrows():
        i = MODEL_NAMES.index(r.src); A, B = r.src, r.dst
        # --- the NULL DETOUR: through A's OWN home regime.  Costs ~0, still burns a
        #     full FT_BUDGET of gradient steps.  If this beats direct, the advantage
        #     is compute, not geometry.
        f_null = floors[(ARCH[A], HOME[A])]
        m_null, c0, _, _, _ = finetune(models[i], KIND[A], data[HOME[A]], f_null)
        _, c0b, _, _, _ = finetune(m_null, KIND[A], data[HOME[B]], floors[(ARCH[A], HOME[B])])
        chains.append(dict(seed=s, src=A, via='__NULL__', dst=B, is_null=True,
                           direct=float(r.d_cost), leg1=float(c0), leg2=float(c0b),
                           chained=float(c0+c0b),
                           advantage=float((r.d_cost-(c0+c0b))/max(r.d_cost, 1e-9)),
                           dominates=bool((c0+c0b) < r.d_cost*(1-DOMINANCE_MARGIN))))
        # --- real detours ---
        for k in range(N_MODELS):
            C = MODEL_NAMES[k]
            if C in (A, B) or HOME[C] in (HOME[A], HOME[B]): continue
            m1, c1, _, _, _ = finetune(models[i], KIND[A], data[HOME[C]], floors[(ARCH[A], HOME[C])])
            _,  c2, _, _, _ = finetune(m1, KIND[A], data[HOME[B]], floors[(ARCH[A], HOME[B])])
            chains.append(dict(seed=s, src=A, via=C, dst=B, is_null=False,
                               direct=float(r.d_cost), leg1=float(c1), leg2=float(c2),
                               chained=float(c1+c2),
                               advantage=float((r.d_cost-(c1+c2))/max(r.d_cost, 1e-9)),
                               dominates=bool((c1+c2) < r.d_cost*(1-DOMINANCE_MARGIN))))
    dc = pd.DataFrame(chains)
    dc.to_csv(f'{OUT}/geodesic_chains_v3_seed{s}.csv', index=False)

    real = dc[~dc.is_null]; null = dc[dc.is_null]
    # triangle-inequality violation rate over the real detours
    tri = float(real.dominates.mean()) if len(real) else np.nan
    null_adv = float(null.advantage.median()) if len(null) else np.nan
    null_dominates = bool(null.dominates.any()) if len(null) else False

    print(f'  [seed {s}] done', flush=True)
    return dict(seed=s, rho=float(rho), rho_sym=float(rho_sym), asymmetry=asym,
                positive_transfer_frac=float(df.positive_transfer.mean()),
                censored_frac=float(df.censored.mean()),
                ref_steps_median=float(np.median(list(ref_steps.values()))),
                n_dominated_pairs=int(real.groupby(['src','dst']).dominates.any().sum()) if len(real) else 0,
                best_advantage=float(real.advantage.max()) if len(real) else 0.0,
                tri_violation_rate=tri,
                null_advantage=null_adv,
                null_dominates=null_dominates)


def main():
    res = [run_seed(s) for s in range(N_SEEDS)]
    d = pd.DataFrame(res); d.to_csv(f'{OUT}/transition_cost_v3_by_seed.csv', index=False)
    L = []
    def say(x): print(x); L.append(x)
    def iqr(x):
        x = np.asarray(x, float); x = x[~np.isnan(x)]
        return 'n/a' if not len(x) else f'{np.median(x):.3f} [{np.percentile(x,25):.3f}, {np.percentile(x,75):.3f}]'

    say('\n================ REGISTERED OUTCOMES (v3) ================\n')

    say('TEST 4 -- Does the geometry predict the cost of reform?')
    say(f'  Spearman rho(d_beh, d_cost), directed  : {iqr(d.rho)}   (bar {RHO_BAR})')
    say(f'  seeds with rho >= {RHO_FLOOR}                : {int((d.rho>=RHO_FLOOR).sum())}/{N_SEEDS}')
    say(f'  rho(d_beh, SYMMETRIZED cost) -- ceiling : {iqr(d.rho_sym)}')
    say(f'  cost asymmetry |A->B vs B->A|          : {iqr(d.asymmetry)}')
    say(f'  positive transfer (beats a CONVERGED    : {iqr(d.positive_transfer_frac)}')
    say(f'    purpose-built model of same capacity)')
    say(f'  reference-floor training steps (median) : {iqr(d.ref_steps_median)}  [audit: the floor is a real floor]')
    say(f'  censored transitions                    : {iqr(d.censored_frac)}')
    ok4 = (np.median(d.rho) >= RHO_BAR) and int((d.rho>=RHO_FLOOR).sum()) >= PASS_BAR
    say(f'  H_predict -> {"PASS: behavioral distance predicts reform cost" if ok4 else "FAIL (null): the geometry is decorative"}')

    say('\nTEST 5 -- Can a detour beat a direct move?  (WITH the null-detour control)')
    say(f'  real detours: seeds with >= 1 dominated pair : {int((d.n_dominated_pairs>0).sum())}/{N_SEEDS} (bar {PASS_BAR})')
    say(f'  real detours: best advantage                 : {iqr(d.best_advantage)}')
    say(f'  triangle-inequality violation rate           : {iqr(d.tri_violation_rate)}   [0 in a metric space]')
    say(f'  NULL detour (via A\'s own home): advantage    : {iqr(d.null_advantage)}   (must tie 0, tol {NULL_TIE_TOL})')
    say(f'  NULL detour dominates in                     : {int(d.null_dominates.sum())}/{N_SEEDS} seeds  (must be 0)')
    ok5_real = int((d.n_dominated_pairs>0).sum()) >= PASS_BAR
    ok5_null = (abs(np.median(d.null_advantage)) < NULL_TIE_TOL) and int(d.null_dominates.sum()) == 0
    if ok5_real and ok5_null:
        say('  H_geodesic -> PASS: detours beat direct moves, and the null detour does NOT.')
        say('     The advantage is PATH STRUCTURE, not gradient steps.  Transition cost')
        say('     violates the triangle inequality: factorization space is not a metric space.')
    elif ok5_real and not ok5_null:
        say('  H_geodesic -> FAIL (ARTIFACT): the null detour also beats direct.  The advantage')
        say('     is bought with extra gradient steps, not with geometry.  Test 5 is a design')
        say('     artifact and must be reported as one.  This is the outcome we most needed to')
        say('     be able to detect, and detecting it is why the control exists.')
    else:
        say('  H_geodesic -> FAIL (null holds): the direct transition is not dominated.')

    say('\nIf cost is asymmetric AND violates the triangle inequality while behavioral')
    say('distance is symmetric and metric, then factorization space is a DIRECTED COST')
    say('STRUCTURE, not a metric space -- the Riemannian framing of exploration 04 was')
    say('wrong from the start, as it warned it might be.  The governance reading:')
    say('reform is DIRECTIONAL and PATH-DEPENDENT.  What an institution costs to leave is')
    say('not what it costs to return to, and the route between two forms can matter more')
    say('than the distance between them.\n')

    open(f'{OUT}/xxiii_cost_summary.txt','w').write('\n'.join(L))
    print(f'Wrote {OUT}/xxiii_cost_summary.txt')

if __name__ == '__main__':
    main()
