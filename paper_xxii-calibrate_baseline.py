#!/usr/bin/env python3
"""
paper_xxii-calibrate_baseline.py
================================
PRE-REGISTRATION SUPPORT, NOT A RESULT.

The 20-seed registered run of the certification crisis failed its admission gate, and
the diagnosis was not the one the gate was written to catch.  The baseline did not
"decay": it COLLAPSED, and it collapsed identically in every condition.  Scripted
agents -- a fixed policy, nothing to unlearn -- went 99.4% -> 58.7% -> 0.0% survival
across the pre / post1 / late windows under NO CRISIS AT ALL.  Cooperation went to
zero everywhere.  The crisis conditions were indistinguishable from the baseline
because all of them were measuring a dead population.

The environment is a ~250-step world.  Resource regrowth cannot sustain three
harvesting agents indefinitely; training episodes are 250 steps and reset the
resources each time, so no policy ever had to face late-episode scarcity.  Evaluating
for 500 steps ran the world past its carrying capacity.  (Paper 13's v3 pilot
evaluated to 400 and its "unstable post2 baseline" was the same thing, one window
earlier.  It was read at the time as a tuning wobble.  It was not.)

WHAT THIS SCRIPT DOES
---------------------
Finds environment parameters under which the NO-CRISIS baseline is STATIONARY over the
full 500-step evaluation horizon -- survival and true-informed giving flat across
windows, not merely nonzero.  It uses SCRIPTED agents only: no DQN, no training, so
it is cheap, and more importantly its outcome cannot be tuned to a policy.

WHAT THIS SCRIPT MUST NOT DO
----------------------------
Look at any crisis condition.  Calibrating an environment against the effect you hope
to find is how a preregistration becomes decoration.  Only `no_crisis` is simulated
here.  The crisis manipulations are not even implemented in this file.

PROCEDURE
---------
1. Sweep REGROWTH_RATE x CONSUME_GAIN with scripted agents, no_crisis, 500 steps.
2. Select the LEAST generous cell that clears the stationarity criterion below.
   Least generous, not best: a world with resources to spare would make cooperation
   unnecessary, and the demonstration needs giving to matter.
3. Freeze those constants into paper_xxii-certification_crisis.py, set the training
   episode length to 500 to match the evaluation horizon, and re-run the registered
   experiment without further tuning.

STATIONARITY CRITERION (registered here, before the sweep)
---------------------------------------------------------
Over 10 scripted seeds under no_crisis:
    (a) median survival_late     >= 0.90 * median survival_pre
    (b) median coop_late         >= 0.70 * median coop_pre
    (c) median true_informed_late >= 0.60          (the original GATE_BAR)
    (d) no seed with survival_late == 0
A cell passes only if all four hold.
"""

import os, random, itertools
import numpy as np
import pandas as pd

# ---------------- sweep grid ----------------
REGROWTH_GRID = [0.12, 0.20, 0.30, 0.45, 0.60]     # 0.12 is the current (failing) value
CONSUME_GRID  = [12.0, 15.0]
N_SEEDS       = 10
EVAL_STEPS    = 500
W_PRE, W_POST1, W_LATE = (0,200), (200,250), (400,500)

# ---------------- world constants (unchanged except where swept) ----------------
GRID_SIZE=5; NUM_AGENTS=3; RESOURCE_CAP=3
ENERGY_MAX=20.0; INVENTORY_CAP=3
GIVE_COST=0.3; METABOLIC_COST=0.4
HARVEST_EFFICIENCY=np.array([[2.,0.],[0.,2.],[1.2,1.2]])
CLOSE_HOMES=[(1,1),(1,3),(2,2)]
DIRS=[(-1,0),(1,0),(0,-1),(0,1)]
INIT_INV=[[2,0],[0,2],[1,1]]


def build_capacity_maps():
    cA=np.ones((GRID_SIZE,GRID_SIZE))*0.05; cB=np.ones((GRID_SIZE,GRID_SIZE))*0.05
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            d0=np.hypot(r-CLOSE_HOMES[0][0], c-CLOSE_HOMES[0][1])
            d1=np.hypot(r-CLOSE_HOMES[1][0], c-CLOSE_HOMES[1][1])
            d2=np.hypot(r-CLOSE_HOMES[2][0], c-CLOSE_HOMES[2][1])
            cA[r,c]+=2.5*np.exp(-d0**2/1.2)-0.3*np.exp(-d1**2/1.2)+0.5*np.exp(-d2**2/2.0)
            cB[r,c]+=2.5*np.exp(-d1**2/1.2)-0.3*np.exp(-d0**2/1.2)+0.5*np.exp(-d2**2/2.0)
    return np.clip(cA,0.01,RESOURCE_CAP), np.clip(cB,0.01,RESOURCE_CAP)


class ScriptedAgent:
    def __init__(self, idx, pos):
        self.idx=idx; self.type=idx; self.pos=pos
        self.energy=15.; self.inventory=np.array(INIT_INV[idx],dtype=float)
        self.signal=np.array([0.,0.])
    def decide(self, w):
        r,c=self.pos
        def adj(d):
            dr,dc=DIRS[d]; nr,nc=r+dr,c+dc
            if 0<=nr<GRID_SIZE and 0<=nc<GRID_SIZE:
                for o in w.agents:
                    if o is not self and o.energy>0 and o.pos==(nr,nc): return o
            return None
        sA=self.inventory[0]>1.; sB=self.inventory[1]>1.
        for d in range(4):
            o=adj(d)
            if o:
                if sA and o.signal[0]>.5 and self.energy>=GIVE_COST: return 8+d
                if sB and o.signal[1]>.5 and self.energy>=GIVE_COST: return 12+d
        if self.energy<10. and self.inventory[0]>=1 and self.inventory[1]>=1: return 7
        if self.type==0:
            if w.resources[r,c,0]>0 and self.inventory[0]<INVENTORY_CAP: return 5
        elif self.type==1:
            if w.resources[r,c,1]>0 and self.inventory[1]<INVENTORY_CAP: return 6
        else:
            if self.inventory[0]<self.inventory[1] and w.resources[r,c,0]>0 and self.inventory[0]<INVENTORY_CAP: return 5
            if w.resources[r,c,1]>0 and self.inventory[1]<INVENTORY_CAP: return 6
            if w.resources[r,c,0]>0 and self.inventory[0]<INVENTORY_CAP: return 5
        tgt=None
        if self.type==0:
            tgt = CLOSE_HOMES[0] if self.inventory[0]<=1. else (CLOSE_HOMES[2] if self.inventory[1]<1. else None)
        elif self.type==1:
            tgt = CLOSE_HOMES[1] if self.inventory[1]<=1. else (CLOSE_HOMES[2] if self.inventory[0]<1. else None)
        else:
            if sA or sB:
                best=float('inf')
                for o in w.agents:
                    if o is not self and o.energy>0 and ((sA and o.signal[0]>.5) or (sB and o.signal[1]>.5)):
                        dd=abs(r-o.pos[0])+abs(c-o.pos[1])
                        if dd<best: best=dd; tgt=o.pos
            if tgt is None: tgt=(2,2)
        valid=[d for d,(dr,dc) in enumerate(DIRS)
               if 0<=r+dr<GRID_SIZE and 0<=c+dc<GRID_SIZE and
               not any(o.energy>0 and o.pos==(r+dr,c+dc) for o in w.agents if o is not self)]
        if tgt and valid:
            return min(valid, key=lambda d: abs(r+DIRS[d][0]-tgt[0])+abs(c+DIRS[d][1]-tgt[1]))
        return random.choice(valid) if valid else 4


class World:
    """no_crisis only.  The crisis manipulations are deliberately not implemented here."""
    def __init__(self, agents, cA, cB, regrowth, consume_gain):
        self.agents=agents; self.cap_A=cA; self.cap_B=cB
        self.regrowth=regrowth; self.consume_gain=consume_gain
        self.resources=np.zeros((GRID_SIZE,GRID_SIZE,2))
        self.resources[:,:,0]=np.random.rand(GRID_SIZE,GRID_SIZE)*cA
        self.resources[:,:,1]=np.random.rand(GRID_SIZE,GRID_SIZE)*cB
        self.pending=[]

    def set_signals(self):
        for a in self.agents:
            if a.energy<=0: a.signal[:]=0.; continue
            a.signal[0]=1. if a.inventory[0]<1. else 0.
            a.signal[1]=1. if a.inventory[1]<1. else 0.

    def regrow(self):
        self.resources[:,:,0]=np.minimum(self.resources[:,:,0]+self.regrowth, self.cap_A)
        self.resources[:,:,1]=np.minimum(self.resources[:,:,1]+self.regrowth, self.cap_B)

    def step(self, t):
        self.set_signals()
        ev=dict(give=0, true=0, surp=0, miss=0)
        for i,j in itertools.combinations(range(NUM_AGENTS),2):
            ai,aj=self.agents[i],self.agents[j]
            if ai.energy<=0 or aj.energy<=0: continue
            if abs(ai.pos[0]-aj.pos[0])+abs(ai.pos[1]-aj.pos[1])!=1: continue
            for giver,taker in ((ai,aj),(aj,ai)):
                for res in (0,1):
                    if giver.inventory[res]>1. and taker.inventory[res]<1.:
                        ev['surp']+=1
                        if taker.signal[res]<0.5: ev['miss']+=1
        order=list(range(NUM_AGENTS)); random.shuffle(order)
        for i in order:
            a=self.agents[i]
            if a.energy<=0: continue
            self._apply(a, a.decide(self), t, ev)
            a.energy=max(0., a.energy-METABOLIC_COST)
        self.regrow()
        return ev

    def _apply(self, a, action, t, ev):
        r,c=a.pos
        if action<4:
            dr,dc=DIRS[action]; nr,nc=r+dr,c+dc
            if 0<=nr<GRID_SIZE and 0<=nc<GRID_SIZE and \
               not any(o.energy>0 and o.pos==(nr,nc) for o in self.agents if o is not a):
                a.pos=(nr,nc)
        elif action in (5,6):
            res=action-5; eff=HARVEST_EFFICIENCY[a.type,res]; avail=self.resources[r,c,res]
            if avail>0 and eff>0:
                h=min(avail,1.); self.resources[r,c,res]-=h
                a.inventory[res]+=min(h*eff, INVENTORY_CAP-a.inventory[res])
        elif action==7:
            if a.inventory[0]>=1 and a.inventory[1]>=1:
                a.inventory-=1
                a.energy=min(a.energy+self.consume_gain, ENERGY_MAX)
        elif 8<=action<=15:
            res=0 if action<12 else 1
            d=action-8 if action<12 else action-12
            dr,dc=DIRS[d]
            for o in self.agents:
                if o is not a and o.energy>0 and o.pos==(r+dr,c+dc):
                    accepted=min(1., INVENTORY_CAP-o.inventory[res])
                    if accepted>0 and a.inventory[res]>=1 and a.energy>=GIVE_COST:
                        true_need = o.inventory[res] < 1.
                        a.inventory[res]-=1; o.inventory[res]+=accepted; a.energy-=GIVE_COST
                        ev['give']+=1
                        if true_need: ev['true']+=1


def run(regrowth, consume_gain, seed):
    np.random.seed(seed); random.seed(seed)
    cA,cB=build_capacity_maps()
    agents=[ScriptedAgent(i, CLOSE_HOMES[i]) for i in range(NUM_AGENTS)]
    w=World(agents,cA,cB,regrowth,consume_gain)
    acc={k: dict(surv=0,give=0,true=0,surp=0,miss=0) for k in ('pre','post1','late')}
    bounds={'pre':W_PRE,'post1':W_POST1,'late':W_LATE}
    for t in range(EVAL_STEPS):
        ev=w.step(t)
        for k,(lo,hi) in bounds.items():
            if lo<=t<hi:
                a_=acc[k]
                a_['surv'] += sum(g.energy>0 for g in agents)
                a_['give'] += ev['give']; a_['true'] += ev['true']
                a_['surp'] += ev['surp']; a_['miss'] += ev['miss']
    out={}
    for k,(lo,hi) in bounds.items():
        a_=acc[k]; n=hi-lo; g=max(a_['give'],1)
        out[f'survival_{k}']      = a_['surv']/(3*n)*100
        out[f'coop_{k}']          = a_['give']
        out[f'true_informed_{k}'] = a_['true']/g
        out[f'uncert_{k}']        = a_['miss']/a_['surp'] if a_['surp']>0 else 0.
    return out


def main():
    rows=[]
    for rg, cg in itertools.product(REGROWTH_GRID, CONSUME_GRID):
        for s in range(N_SEEDS):
            r=run(rg,cg,s); r.update(regrowth=rg, consume_gain=cg, seed=s)
            rows.append(r)
    d=pd.DataFrame(rows)
    os.makedirs('xxii_out', exist_ok=True)
    d.to_csv('xxii_out/baseline_calibration.csv', index=False)

    print(f'{"regrow":>7} {"gain":>5} | {"surv_pre":>9} {"surv_late":>10} '
          f'{"coop_pre":>9} {"coop_late":>10} {"trueinf_late":>13} | pass')
    print('-'*82)
    passing=[]
    for (rg,cg), g in d.groupby(['regrowth','consume_gain']):
        sp, sl = g.survival_pre.median(), g.survival_late.median()
        cp, cl = g.coop_pre.median(),     g.coop_late.median()
        ti     = g.true_informed_late.median()
        ok = (sl >= 0.90*sp) and (cl >= 0.70*cp) and (ti >= 0.60) and (g.survival_late > 0).all()
        if ok: passing.append((rg,cg,sp,sl,cp,cl,ti))
        print(f'{rg:7.2f} {cg:5.1f} | {sp:9.1f} {sl:10.1f} {cp:9.1f} {cl:10.1f} {ti:13.3f} | '
              f'{"PASS" if ok else "fail"}')

    print()
    if not passing:
        print('NO CELL PASSES.  The environment cannot sustain a stationary cooperative')
        print('equilibrium over 500 steps under ANY swept parameter.  In that case the')
        print('demonstration must be redesigned rather than retuned -- either shorten the')
        print('horizon (which forecloses the recovery-window test, C3, and therefore the')
        print('only new claim Paper XXII had) or rebuild the resource economy.')
        print('Report this; do not widen the grid until something passes.')
        return

    # least generous passing cell: cooperation must still be necessary
    rg,cg,sp,sl,cp,cl,ti = sorted(passing, key=lambda x: (x[0], x[1]))[0]
    print(f'SELECTED (least generous passing cell): REGROWTH_RATE={rg}, CONSUME_GAIN={cg}')
    print(f'  survival {sp:.1f}% -> {sl:.1f}%   cooperation {cp:.0f} -> {cl:.0f}   '
          f'true-informed (late) {ti:.3f}')
    print()
    print('Freeze these into paper_xxii-certification_crisis.py, set the training episode')
    print('length to 500 to match the evaluation horizon, and re-run the registered')
    print('experiment WITHOUT further tuning.  Note in the paper that the environment was')
    print('calibrated on the no-crisis baseline only, before the crisis conditions were')
    print('run, and that the first registered run failed its gate and is reported as a fail.')

if __name__ == '__main__':
    main()
