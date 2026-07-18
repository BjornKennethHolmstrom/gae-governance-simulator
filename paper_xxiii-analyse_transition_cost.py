#!/usr/bin/env python3
"""
paper_xxiii-analyse_transition_cost.py
======================================
REGISTERED ANALYSIS.  Committed BEFORE the full run of paper_xxiii_transition_cost_v4.py.

Reads the per-seed CSVs that v4 writes and computes the registered outcomes.  It is
separate from the generating script so that the analysis is fixed in advance and can
be diffed against this file.  The v4 smoke run (2 seeds) motivated two of the claims
below; they are therefore REGISTERED HERE, before the full run, rather than harvested
from it.

REGISTERED CLAIMS
-----------------
H_predict   Behavioral distance predicts reform cost.
            rho(d_beh, d_cost) >= RHO_BAR median across seeds, >= RHO_FLOOR in >= PASS_BAR.
            NULL: the geometry is decorative.

H_geodesic  The ROUTE matters, and not because of gradient steps.
            The best real detour beats the NULL detour (routing through the source's
            own home regime -- same budget, same structure, no destination) by
            >= DOMINANCE_MARGIN, in >= PASS_BAR seeds.
            NULL: all intermediates are interchangeable; the gain is compute.

H_route     A BADLY chosen route is worse than no route.
            (worst - best) / null >= ROUTE_SPREAD_BAR, and the worst real detour
            loses to the null in a majority of pairs.
            NULL: the choice of intermediate is immaterial.

H_hard      *** NEW, registered before the full run ***
            THE ROUTE MATTERS IN PROPORTION TO HOW HARD THE REFORM IS.
            rho(direct cost, gain of best-real over null) >= HARD_RHO across pairs.
            NULL: routes matter equally on easy and hard reforms -- in which case
            "choose your path" is advice without a domain.

            Motivation (v4 smoke, 2 seeds, 12 pairs): every winning detour had a
            source with an expensive direct transition (damped_h8, direct cost
            8.3-8.8), and every losing detour had a cheap one (null leg2 of
            0.68-2.22).  On easy reforms, detouring is strictly worse than going
            direct.  This is the claim an institution would actually act on, and it
            is registered rather than reported.

H_waypoint  *** NEW, registered before the full run ***
            A REAL WAYPOINT SHORTENS THE ONWARD LEG; A FREE ONE DOES NOT.
            Among winning detours, leg2 is cut relative to the null's leg2 by
            >= LEG2_CUT_BAR.  Among losing detours, it is not.
            AND: the "hub" is not merely the cheapest regime to enter --
            best_via == argmin(leg1) in FEWER than TRIVIAL_HUB_BAR of pairs.

            Motivation (v4 smoke): best_via was the cheapest-to-enter in 75% of
            pairs, so a naive "hub" reading is mostly trivial.  What is NOT trivial:
            winners cut leg2 by 60-96% and had a NONZERO entry cost, while losers
            had a FREE entry and INCREASED leg2 by 45-310%.  A free entry means the
            model never actually went anywhere -- it already performed at the
            intermediate's level -- so the gradient steps moved it somewhere useless.

            NULL: leg2 reduction does not distinguish winners from losers, and/or
            best_via is simply the cheapest entry -- in which case "waypoint" means
            nothing more than "the easy regime".

PRE-DECLARED SENSITIVITY
------------------------
compressed_h2 (hidden dim 2) is a degenerate source: its transition costs are
dominated by CAPACITY, not by path (smoke: leg2 of 10.9 and 23.3, an order of
magnitude above the rest).  Every outcome is reported BOTH with and without it as a
source.  Declared before the run; not a filter applied after seeing results.
"""

import sys, glob, itertools, collections
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ======================================================================
# REGISTERED CONSTANTS
# ======================================================================
PASS_BAR         = 8      # of N_SEEDS = 10
RHO_BAR          = 0.50
RHO_FLOOR        = 0.30
DOMINANCE_MARGIN = 0.10
ROUTE_SPREAD_BAR = 0.30
HARD_RHO         = 0.40   # H_hard
LEG2_CUT_BAR     = 0.40   # H_waypoint: winners must cut the onward leg by >= 40%
TRIVIAL_HUB_BAR  = 0.80   # H_waypoint: if best_via == cheapest entry MORE often than
                          # this, the hub is trivial and the waypoint claim is withdrawn

OUT = 'xxiii_out'
DEGENERATE_SOURCE = 'compressed_h2'


def load():
    costs  = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{OUT}/transition_costs_v4_seed*.csv'))])
    chains = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{OUT}/geodesic_chains_v4_seed*.csv'))])
    return costs, chains


def pair_table(chains):
    """One row per (seed, src, dst): the null detour, the best and worst real detour."""
    rows = []
    for (s, src, dst), g in chains.groupby(['seed', 'src', 'dst']):
        nl = g[g.is_null]; rl = g[~g.is_null]
        if not len(nl) or not len(rl): continue
        null_c   = float(nl.chained.iloc[0])
        null_l2  = float(nl.leg2.iloc[0])       # == the direct move, after a null warm-up
        direct   = float(nl.direct.iloc[0])
        best     = rl.loc[rl.chained.idxmin()]
        worst    = rl.loc[rl.chained.idxmax()]
        cheapest = rl.loc[rl.leg1.idxmin()]
        rows.append(dict(
            seed=s, src=src, dst=dst, direct=direct, null=null_c, null_leg2=null_l2,
            best_via=best.via, best=float(best.chained),
            best_leg1=float(best.leg1), best_leg2=float(best.leg2),
            worst=float(worst.chained), worst_via=worst.via,
            cheapest_entry_via=cheapest.via,
            best_is_cheapest_entry=bool(best.via == cheapest.via),
            gain_vs_null=(null_c - float(best.chained)) / max(null_c, 1e-9),
            beats_null=bool(float(best.chained) < null_c * (1 - DOMINANCE_MARGIN)),
            route_spread=(float(worst.chained) - float(best.chained)) / max(null_c, 1e-9),
            worst_loses_to_null=bool(float(worst.chained) > null_c),
            leg2_cut=1.0 - float(best.leg2) / max(null_l2, 1e-9),
            compute_component=(direct - null_c) / max(direct, 1e-9),
            total_advantage=(direct - float(best.chained)) / max(direct, 1e-9),
        ))
    return pd.DataFrame(rows)


def report(costs, pairs, label, say):
    seeds = sorted(costs.seed.unique()); n = len(seeds)

    def iqr(x):
        x = np.asarray(x, float); x = x[~np.isnan(x)]
        return 'n/a' if not len(x) else f'{np.median(x):.3f} [{np.percentile(x,25):.3f}, {np.percentile(x,75):.3f}]'

    say(f'\n=============== {label} ===============\n')

    # ---------------- H_predict ----------------
    rhos, rho_syms, asyms = [], [], []
    for s in seeds:
        c = costs[costs.seed == s]
        rhos.append(spearmanr(c.d_beh, c.d_cost)[0])
        m = {(r.src, r.target_regime): r.d_cost for r in c.itertuples()}
        h = {(r.src, r.target_regime): r.d_beh for r in c.itertuples()}
        srcs = sorted(c.src.unique()); sym, beh, asy = [], [], []
        for A, B in itertools.combinations(srcs, 2):
            RA = c[c.src == A].target_regime.tolist(); RB = c[c.src == B].target_regime.tolist()
            ab = next((m[(A, r)] for r in RB if (A, r) in m and r not in RA), None)
            ba = next((m[(B, r)] for r in RA if (B, r) in m and r not in RB), None)
            hb = next((h[(A, r)] for r in RB if (A, r) in h and r not in RA), None)
            if ab is None or ba is None or hb is None: continue
            sym.append((ab+ba)/2); beh.append(hb)
            asy.append(abs(ab-ba)/max(ab, ba, 1e-9))
        rho_syms.append(spearmanr(beh, sym)[0] if len(beh) > 2 else np.nan)
        asyms.append(np.median(asy) if asy else np.nan)

    say('H_predict -- does behavioral distance predict reform cost?')
    say(f'  rho(d_beh, d_cost), directed        : {iqr(rhos)}   (bar {RHO_BAR})')
    say(f'  rho(d_beh, symmetrized cost) ceiling: {iqr(rho_syms)}')
    say(f'  cost asymmetry |A->B vs B->A|       : {iqr(asyms)}   [distance is symmetric; cost is NOT]')
    say(f'  censored transitions                : {iqr(costs.groupby("seed").censored.mean())}')
    say(f'  positive transfer vs converged floor: {iqr(costs.groupby("seed").positive_transfer.mean())}')
    ok = np.median(rhos) >= RHO_BAR and sum(r >= RHO_FLOOR for r in rhos) >= min(PASS_BAR, n)
    say(f'  -> {"PASS" if ok else "FAIL (null): the geometry is decorative"}\n')

    if not len(pairs):
        say('  (no chain data)'); return

    per = pairs.groupby('seed')

    # ---------------- H_geodesic ----------------
    frac = per.beats_null.mean(); gain = per.gain_vs_null.median()
    say('H_geodesic -- does the ROUTE matter, at EQUAL COMPUTE?')
    say(f'  pairs where best real detour beats the NULL : {iqr(frac)}')
    say(f'  median gain of best-real OVER the null      : {iqr(gain)}   (bar {DOMINANCE_MARGIN})')
    say(f'  compute component (null vs direct)          : {iqr(per.compute_component.median())}  <- gradient steps alone')
    say(f'  total advantage   (best-real vs direct)     : {iqr(per.total_advantage.median())}')
    nb = int((frac >= 0.5).sum())
    okg = nb >= min(PASS_BAR, n) and np.median(gain) >= DOMINANCE_MARGIN
    say(f'  -> {nb}/{n} seeds. {"PASS: the advantage is PATH STRUCTURE, not compute" if okg else "FAIL (null): all intermediates are interchangeable"}\n')

    # ---------------- H_route ----------------
    spread = per.route_spread.median(); wl = per.worst_loses_to_null.mean()
    say('H_route -- is a BADLY chosen route worse than no route?')
    say(f'  (worst - best) / null                       : {iqr(spread)}   (bar {ROUTE_SPREAD_BAR})')
    say(f'  pairs whose WORST real detour loses to null : {iqr(wl)}')
    okr = np.median(spread) >= ROUTE_SPREAD_BAR and np.median(wl) > 0.5
    say(f'  -> {"PASS: route choice matters, and a bad route is worse than no route" if okr else "FAIL (null): the intermediate is immaterial"}\n')

    # ---------------- H_hard (NEW) ----------------
    rh = []
    for s in seeds:
        p = pairs[pairs.seed == s]
        if len(p) > 3: rh.append(spearmanr(p.direct, p.gain_vs_null)[0])
    say('H_hard -- does the route matter MORE on HARDER reforms?')
    say(f'  rho(direct cost, gain of best-real over null): {iqr(rh)}   (bar {HARD_RHO})')
    okh = len(rh) > 0 and np.median(rh) >= HARD_RHO
    say(f'  -> {"PASS: routes matter in proportion to reform difficulty. On EASY reforms, detouring is a WASTE." if okh else "FAIL (null): routes matter equally regardless of difficulty"}\n')

    # ---------------- H_waypoint (NEW) ----------------
    win  = pairs[pairs.beats_null]
    lose = pairs[~pairs.beats_null]
    triv = pairs.best_is_cheapest_entry.mean()
    say('H_waypoint -- does a real waypoint SHORTEN the onward leg?')
    say(f'  leg2 cut, among WINNING detours  : {iqr(win.leg2_cut)}   (bar {LEG2_CUT_BAR})')
    say(f'  leg2 cut, among LOSING detours   : {iqr(lose.leg2_cut)}   [expected: <= 0, i.e. leg2 grows]')
    say(f'  entry cost (leg1), winners       : {iqr(win.best_leg1)}')
    say(f'  entry cost (leg1), losers        : {iqr(lose.best_leg1)}   [expected: ~0 -- a FREE entry goes nowhere]')
    say(f'  best_via == cheapest to enter    : {triv:.0%}   (trivial if > {TRIVIAL_HUB_BAR:.0%})')
    okw = (len(win) > 0 and np.median(win.leg2_cut) >= LEG2_CUT_BAR
           and (not len(lose) or np.median(lose.leg2_cut) < LEG2_CUT_BAR)
           and triv <= TRIVIAL_HUB_BAR)
    say(f'  -> {"PASS: a real waypoint shortens the onward leg; a FREE entry does not move you at all" if okw else "FAIL: the waypoint is trivial (just the cheapest regime) or does not shorten the onward leg"}\n')

    hubs = collections.Counter(win.best_via)
    if hubs:
        say('  waypoints chosen (among winners): ' + ', '.join(f'{k} x{v}' for k, v in hubs.most_common(4)))
    say('')


def main():
    costs, chains = load()
    pairs = pair_table(chains)
    L = []
    def say(x): print(x); L.append(x)

    say('======== PAPER XXIII: REGISTERED TRANSITION-COST OUTCOMES ========')
    report(costs, pairs, 'ALL SOURCES', say)

    say('\n' + '='*66)
    say(f'PRE-DECLARED SENSITIVITY: excluding {DEGENERATE_SOURCE} as a source.')
    say('(hidden dim 2 -- its costs are capacity-dominated, not path-dominated)')
    say('='*66)
    report(costs[costs.src != DEGENERATE_SOURCE],
           pairs[pairs.src != DEGENERATE_SOURCE],
           f'EXCLUDING {DEGENERATE_SOURCE}', say)

    say('\nIf the two panels disagree, the disagreement IS the result: the effect is')
    say('capacity-driven rather than path-driven, and the paper says so.')

    open(f'{OUT}/xxiii_cost_registered.txt', 'w').write('\n'.join(L))
    print(f'\nWrote {OUT}/xxiii_cost_registered.txt')

if __name__ == '__main__':
    main()
