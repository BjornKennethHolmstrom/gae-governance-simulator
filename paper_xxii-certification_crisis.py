#!/usr/bin/env python3
"""
paper_xxii-certification_crisis.py  (v2)
========================================
Registered demonstration for Paper XXII, Section 4.

WHY THERE IS A v2: THE FIRST REGISTERED RUN FAILED ITS GATE, AND IT WAS RIGHT TO
--------------------------------------------------------------------------------
The 20-seed run reported no_crisis late-window true-informed giving of 0.000
[0.000, 0.000], and the gate withheld C1-C4.  The diagnosis was not decay.  Scripted
agents -- a fixed policy with nothing to unlearn -- went 99.4% -> 58.7% -> 0.0%
survival across the pre/post1/late windows under NO CRISIS AT ALL, and every crisis
condition returned identical medians because all of them were measuring a dead
population.  Baseline calibration then showed the collapse was insensitive to a 5x
change in resource regrowth, which ruled out scarcity.

The cause was a degenerate action.  Harvest succeeded whenever
`resources[r,c,res] > 0`, and because the capacity map is clipped at a floor of 0.01
and regrows every step, that condition is true on EVERY cell, ALWAYS.  So an agent
that drifted onto a barren cell could harvest it forever, scraping ~0.01 units, and
never travel home.  Tracing confirmed it: the A-specialist spent its last 120 steps
parked on cell (1,4) -- A-capacity 0.01 -- harvesting 112 times, and starved there
while the grid sat saturated.  The DQN had the same attractor by a different route:
harvesting paid `add*0.1` and cost nothing, so scraping a dead cell weakly dominated
moving.  Over a 250-step training episode the initial inventory buffer hides all this.
Over a 500-step evaluation it kills everyone.

This is why Paper 13's v3 pilot had an "unstable post2 baseline", which was read at the
time as a tuning wobble.  It was not a wobble.  It was this, one window earlier.

WHAT CHANGED
------------
1. HARVEST_MIN (world rule, applies to every agent type).  Harvest requires a cell
   holding at least HARVEST_MIN of the resource.  Below that the action fails and
   costs a little.  This is a RULE fix, not a PARAMETER fix: regrowth and consume gain
   are left at their original values (0.12, 12.0), because baseline calibration showed
   the world is stationary once the degenerate action is removed.  Nothing was dialled
   toward an outcome.

2. TRAIN_STEPS = EVAL_STEPS = 500.  The policy was trained on 250-step episodes and
   evaluated over 500, so the late window -- the window the recovery test depends on --
   was entirely out of distribution.

3. The stationarity gate is fixed and widened.  The v1 gate compared cooperation as a
   RAW COUNT across windows of unequal length (200 vs 100 steps), which understated
   late-window cooperation by a factor of two.  It is now a rate, and the gate checks
   survival, cooperation rate, and true-informed giving.

CARRIED OVER FROM v1
--------------------
Per-seed retraining (phenomenon-level, not model-identity).  Reset-delay sweep.
cert_crisis_unused_channel as a registered specificity control.  eps = 0 at evaluation.

NOTE FOR THE SERIES: the harvest defect lives in the shared grid-world environment, so
it also affects the 12-simulation* coordination runs that will seed the multi-agent
series.  Their 250-step episodes mask it.  It should be fixed there before any of that
material is used.  Papers XIX/XX/XXI use the bouncing-dot zoo and are unaffected.

REGISTERED PREDICTIONS: unchanged from v1 (GATE, C1-C4).  See below.
"""

import os, csv, random, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from multiprocessing import Pool
from scipy.stats import spearmanr

# ======================================================================
# REGISTERED CONSTANTS
# ======================================================================
N_SEEDS      = 20
PASS_BAR     = 16
GATE_BAR     = 0.60        # no_crisis late-window true-informed-give rate
GATE_SURV    = 0.90        # survival_late >= GATE_SURV * survival_pre
GATE_COOP    = 0.70        # coop RATE late >= GATE_COOP * coop RATE pre
C1_MARGIN    = 0.10
C2_BAR       = 15
C3_RHO       = -0.50
C3_NULLDIFF  = 0.10
C4_TOL       = 0.10
RESET_DELAYS = [10, 25, 50, 100]

TRAIN_EPISODES = 600
TRAIN_STEPS    = 500       # was 250 -- must match the evaluation horizon
EVAL_STEPS     = 500
CRISIS_STEP    = 200
W_PRE   = (0,   200)
W_POST1 = (200, 250)
W_LATE  = (400, 500)
WIN_LEN = {'pre': 200, 'post1': 50, 'late': 100}   # for rate normalisation

OUT_DIR = 'xxii_out'; os.makedirs(OUT_DIR, exist_ok=True)

# ---------- World config ----------
GRID_SIZE=5; NUM_AGENTS=3; RESOURCE_CAP=3; REGROWTH_RATE=0.12    # unchanged
ENERGY_MAX=20.0; INVENTORY_CAP=3; CONSUME_GAIN=12.0              # unchanged
GIVE_COST=0.3; METABOLIC_COST=0.4; GIVER_CREDIT_REWARD=2.0
HARVEST_MIN=0.5            # NEW: a cell must actually hold something to be harvested
HARVEST_FAIL_COST=0.05     # NEW: scraping a barren cell is no longer free
HARVEST_EFFICIENCY = np.array([[2.,0.],[0.,2.],[1.2,1.2]])
CLOSE_HOMES=[(1,1),(1,3),(2,2)]
DIRS=[(-1,0),(1,0),(0,-1),(0,1)]
INIT_INV=[[2,0],[0,2],[1,1]]

CONDITIONS = (['no_crisis', 'ordinary_disturbance',
               'cert_crisis_used_channel', 'cert_crisis_unused_channel']
              + [f'reset_d{d}' for d in RESET_DELAYS])

METRICS = ['survival','coop','apparent_informed','true_informed',
           'false_certified','certification_error','uncertified_true_need',
           'strict_usefulness','reciprocity']
WINDOWS = ['pre','post1','late']
CSV_FIELDS = ['condition','agent_type','seed'] + \
             [f'{m}_{w}' for m in METRICS for w in WINDOWS]

# ======================================================================
def build_capacity_maps():
    cap_A = np.ones((GRID_SIZE,GRID_SIZE))*0.05
    cap_B = np.ones((GRID_SIZE,GRID_SIZE))*0.05
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            d0=np.hypot(r-CLOSE_HOMES[0][0], c-CLOSE_HOMES[0][1])
            d1=np.hypot(r-CLOSE_HOMES[1][0], c-CLOSE_HOMES[1][1])
            d2=np.hypot(r-CLOSE_HOMES[2][0], c-CLOSE_HOMES[2][1])
            cap_A[r,c]+=2.5*np.exp(-d0**2/1.2)-0.3*np.exp(-d1**2/1.2)+0.5*np.exp(-d2**2/2.0)
            cap_B[r,c]+=2.5*np.exp(-d1**2/1.2)-0.3*np.exp(-d0**2/1.2)+0.5*np.exp(-d2**2/2.0)
    return np.clip(cap_A,0.01,RESOURCE_CAP), np.clip(cap_B,0.01,RESOURCE_CAP)

class DQN(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i,128), nn.ReLU(),
                                 nn.Linear(128,128), nn.ReLU(), nn.Linear(128,o))
    def forward(self,x): return self.net(x)

class DQNAgent:
    def __init__(self, idx, atype, pos, in_dim, act_dim):
        self.idx=idx; self.type=atype; self.pos=pos
        self.energy=15.; self.inventory=np.array(INIT_INV[idx],dtype=float)
        self.signal=np.array([0.,0.]); self.memory=np.zeros((NUM_AGENTS,6))
        self.q_network=DQN(in_dim,act_dim); self.target_network=DQN(in_dim,act_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer=optim.Adam(self.q_network.parameters(), lr=1e-3)
        self.replay_buffer=deque(maxlen=5000)
        self.epsilon=1.0; self.epsilon_min=0.02; self.epsilon_decay=0.997
        self.gamma=0.95; self.batch_size=64
        self.update_target_every=100; self.step_count=0
    def act(self, obs):
        if random.random() < self.epsilon:
            return random.randrange(self.q_network.net[-1].out_features)
        with torch.no_grad():
            return int(torch.argmax(self.q_network(torch.FloatTensor(obs).unsqueeze(0))).item())
    def remember(self,*t): self.replay_buffer.append(t)
    def replay(self):
        if len(self.replay_buffer) < self.batch_size: return
        b=random.sample(self.replay_buffer, self.batch_size)
        ob=torch.FloatTensor(np.array([t[0] for t in b]))
        ac=torch.LongTensor(np.array([t[1] for t in b])).unsqueeze(1)
        rw=torch.FloatTensor(np.array([t[2] for t in b]))
        nb=torch.FloatTensor(np.array([t[3] for t in b]))
        dn=torch.FloatTensor(np.array([t[4] for t in b]))
        q=self.q_network(ob).gather(1,ac).squeeze()
        with torch.no_grad():
            tgt = rw + self.gamma*self.target_network(nb).max(1)[0]*(1-dn)
        loss=nn.MSELoss()(q, tgt.detach())
        self.optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.)
        self.optimizer.step(); self.step_count+=1
        self.epsilon=max(self.epsilon_min, self.epsilon*self.epsilon_decay)
        if self.step_count % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    def decay_memory(self, d=0.95): self.memory *= d

class ScriptedAgent:
    def __init__(self, idx, atype, pos):
        self.idx=idx; self.type=atype; self.pos=pos
        self.energy=15.; self.inventory=np.array(INIT_INV[idx],dtype=float)
        self.signal=np.array([0.,0.]); self.memory=np.zeros((NUM_AGENTS,6))
    def decay_memory(self, d=0.95): self.memory *= d
    def decide(self, world):
        r,c=self.pos
        def adj(d):
            dr,dc=DIRS[d]; nr,nc=r+dr,c+dc
            if 0<=nr<GRID_SIZE and 0<=nc<GRID_SIZE:
                for o in world.agents:
                    if o is not self and o.energy>0 and o.pos==(nr,nc): return o
            return None
        sA=self.inventory[0]>1.; sB=self.inventory[1]>1.
        for d in range(4):
            o=adj(d)
            if o:
                if sA and o.signal[0]>.5 and self.energy>=GIVE_COST: return 8+d
                if sB and o.signal[1]>.5 and self.energy>=GIVE_COST: return 12+d
        if self.energy<10. and self.inventory[0]>=1 and self.inventory[1]>=1: return 7
        # HARVEST_MIN: do not scrape a barren cell -- go somewhere that has something
        if self.type==0:
            if world.resources[r,c,0]>=HARVEST_MIN and self.inventory[0]<INVENTORY_CAP: return 5
        elif self.type==1:
            if world.resources[r,c,1]>=HARVEST_MIN and self.inventory[1]<INVENTORY_CAP: return 6
        else:
            if self.inventory[0]<self.inventory[1] and world.resources[r,c,0]>=HARVEST_MIN and self.inventory[0]<INVENTORY_CAP: return 5
            if world.resources[r,c,1]>=HARVEST_MIN and self.inventory[1]<INVENTORY_CAP: return 6
            if world.resources[r,c,0]>=HARVEST_MIN and self.inventory[0]<INVENTORY_CAP: return 5
        tgt=None
        if self.type==0:
            tgt = CLOSE_HOMES[0] if self.inventory[0]<=1. else (CLOSE_HOMES[2] if self.inventory[1]<1. else None)
        elif self.type==1:
            tgt = CLOSE_HOMES[1] if self.inventory[1]<=1. else (CLOSE_HOMES[2] if self.inventory[0]<1. else None)
        else:
            if sA or sB:
                best=float('inf')
                for o in world.agents:
                    if o is not self and o.energy>0 and ((sA and o.signal[0]>.5) or (sB and o.signal[1]>.5)):
                        d=abs(r-o.pos[0])+abs(c-o.pos[1])
                        if d<best: best=d; tgt=o.pos
            if tgt is None: tgt=(2,2)
        valid=[d for d,(dr,dc) in enumerate(DIRS)
               if 0<=r+dr<GRID_SIZE and 0<=c+dc<GRID_SIZE and
               not any(o.energy>0 and o.pos==(r+dr,c+dc) for o in world.agents if o is not self)]
        if tgt and valid:
            return min(valid, key=lambda d: abs(r+DIRS[d][0]-tgt[0])+abs(c+DIRS[d][1]-tgt[1]))
        return random.choice(valid) if valid else 4

# ======================================================================
class World:
    def __init__(self, agents, cap_A, cap_B, condition='no_crisis'):
        self.agents=agents; self.cap_A=cap_A; self.cap_B=cap_B
        self.condition=condition
        self.reset_step = (CRISIS_STEP + int(condition.split('_d')[1])
                           if condition.startswith('reset_d') else None)
        self.resources=np.zeros((GRID_SIZE,GRID_SIZE,2)); self.reset_resources()
        self.pending_gifts=[]; self.gift_log=[]; self.flipped=False

    def reset_resources(self):
        self.resources[:,:,0]=np.random.rand(GRID_SIZE,GRID_SIZE)*self.cap_A
        self.resources[:,:,1]=np.random.rand(GRID_SIZE,GRID_SIZE)*self.cap_B

    def regrow(self, step):
        f = 0.5 if (self.condition=='ordinary_disturbance'
                    and CRISIS_STEP <= step < CRISIS_STEP+100) else 1.0
        self.resources[:,:,0]=np.minimum(self.resources[:,:,0]+REGROWTH_RATE*f, self.cap_A)
        self.resources[:,:,1]=np.minimum(self.resources[:,:,1]+REGROWTH_RATE*f, self.cap_B)

    def set_signals(self, step):
        crisis = (self.condition.startswith('cert_crisis')
                  or self.condition.startswith('reset_d'))
        if crisis and step >= CRISIS_STEP: self.flipped = True
        if self.reset_step is not None and step >= self.reset_step: self.flipped = False
        for a in self.agents:
            if a.energy <= 0: a.signal[:] = 0.; continue
            if self.flipped and a.idx == 1:
                if self.condition == 'cert_crisis_unused_channel':
                    a.signal[0] = 1. if a.inventory[0] < 1. else 0.
                    a.signal[1] = 1. if a.inventory[1] >= 2. else 0.
                else:
                    a.signal[0] = 1. if a.inventory[0] >= 1. else 0.
                    a.signal[1] = 1. if a.inventory[1] < 1. else 0.
            else:
                a.signal[0] = 1. if a.inventory[0] < 1. else 0.
                a.signal[1] = 1. if a.inventory[1] < 1. else 0.

    def get_observation(self, i):
        a=self.agents[i]; r,c=a.pos
        own=np.array([a.energy/ENERGY_MAX, a.inventory[0]/INVENTORY_CAP, a.inventory[1]/INVENTORY_CAP])
        loc=self.resources[r,c]/RESOURCE_CAP
        adj=np.zeros(24)
        for d,(dr,dc) in enumerate(DIRS):
            nr,nc=r+dr,c+dc
            if 0<=nr<GRID_SIZE and 0<=nc<GRID_SIZE:
                for o in self.agents:
                    if o is not a and o.energy>0 and o.pos==(nr,nc):
                        b=d*6; adj[b]=1.; adj[b+1+o.idx]=1.
                        adj[b+4]=o.signal[0]; adj[b+5]=o.signal[1]
        mem=a.memory.flatten(); mask=np.ones(18,bool); mask[i*6:(i+1)*6]=False
        oh=np.zeros(3); oh[a.type]=1.
        return np.concatenate([own, loc, adj, mem[mask], oh])

    def step(self, step_num, train=False, track=False, transitions=None, agent_class='dqn'):
        self.set_signals(step_num)
        ev = dict(give_success=0, app_inf=0, true_inf=0, false_cert=0,
                  surp_need=0, miss_cert=0, give_pairs=[], cons_strict=[]) if track else None
        if track:
            for i,j in itertools.combinations(range(NUM_AGENTS), 2):
                ai,aj=self.agents[i],self.agents[j]
                if ai.energy<=0 or aj.energy<=0: continue
                if abs(ai.pos[0]-aj.pos[0])+abs(ai.pos[1]-aj.pos[1]) != 1: continue
                for (giver,taker) in ((ai,aj),(aj,ai)):
                    for res in (0,1):
                        if giver.inventory[res]>1. and taker.inventory[res]<1.:
                            ev['surp_need']+=1
                            if taker.signal[res]<0.5: ev['miss_cert']+=1
        order=list(range(NUM_AGENTS)); random.shuffle(order)
        for i in order:
            a=self.agents[i]
            if a.energy<=0: continue
            if agent_class=='dqn':
                obs=self.get_observation(i); action=a.act(obs)
            else:
                obs=None; action=a.decide(self)
            cur = len(transitions) if transitions is not None else -1
            reward = self._apply(i, action, step_num, cur, ev)
            a.energy=max(0., a.energy-METABOLIC_COST)
            if agent_class=='dqn' and train and transitions is not None:
                transitions.append(dict(agent_idx=i, step=step_num, obs=obs, action=action,
                                        reward=reward, next_obs=self.get_observation(i),
                                        done=a.energy<=0))
        for a in self.agents: a.decay_memory()
        self.regrow(step_num)
        return ev

    def _give(self, a, tgt, res, step_num, cur, ev):
        accepted = min(1., INVENTORY_CAP - tgt.inventory[res])
        if accepted<=0 or a.inventory[res]<1 or a.energy<GIVE_COST: return -0.1
        true_need = tgt.inventory[res] < 1.
        app_need  = tgt.signal[res] > 0.5
        a.inventory[res]-=1; tgt.inventory[res]+=accepted
        a.energy-=GIVE_COST
        self.pending_gifts.append(dict(receiver=tgt.idx, resource=res, step=step_num,
                                       giver=a.idx, needed=true_need))
        if cur >= 0:
            self.gift_log.append(dict(giver=a.idx, receiver=tgt.idx, step=step_num,
                                      transition_index=cur, needed=true_need))
        if ev is not None:
            ev['give_success']+=1; ev['give_pairs'].append((a.idx, tgt.idx))
            if app_need: ev['app_inf']+=1
            if true_need: ev['true_inf']+=1
            if app_need and not true_need: ev['false_cert']+=1
        return -GIVE_COST

    def _apply(self, i, action, step_num, cur, ev):
        a=self.agents[i]; r,c=a.pos; reward=0.
        if action<4:
            dr,dc=DIRS[action]; nr,nc=r+dr,c+dc
            if 0<=nr<GRID_SIZE and 0<=nc<GRID_SIZE and \
               not any(o.energy>0 and o.pos==(nr,nc) for o in self.agents if o is not a):
                a.pos=(nr,nc)
        elif action==4: pass
        elif action in (5,6):
            res=action-5; eff=HARVEST_EFFICIENCY[a.type,res]; avail=self.resources[r,c,res]
            if eff<=0:
                reward-=0.1                       # wrong resource for this specialisation
            elif avail < HARVEST_MIN:
                reward-=HARVEST_FAIL_COST         # NEW: the barren-cell attractor is closed
            elif a.inventory[res] < INVENTORY_CAP:
                h=min(avail,1.); self.resources[r,c,res]-=h
                add=min(h*eff, INVENTORY_CAP-a.inventory[res]); a.inventory[res]+=add
                reward+=add*0.1
        elif action==7:
            if a.inventory[0]>=1 and a.inventory[1]>=1:
                a.inventory-=1
                a.energy=min(a.energy+CONSUME_GAIN, ENERGY_MAX); reward+=CONSUME_GAIN
                if ev is not None:
                    strict=[g for g in self.pending_gifts
                            if g['receiver']==i and step_num-g['step']<=10 and g['needed']]
                    ev['cons_strict'].extend(strict)
                self.pending_gifts=[g for g in self.pending_gifts
                                    if g['receiver']!=i and step_num-g['step']<=10]
            else: reward-=0.5
        elif 8<=action<=15:
            res = 0 if action<12 else 1
            d = action-8 if action<12 else action-12
            dr,dc=DIRS[d]; tgt=None
            for o in self.agents:
                if o is not a and o.energy>0 and o.pos==(r+dr,c+dc): tgt=o
            if tgt is not None: reward += self._give(a, tgt, res, step_num, cur, ev)
        return reward

# ======================================================================
def train_dqn(seed):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    cap_A,cap_B=build_capacity_maps()
    agents=[DQNAgent(i,i,CLOSE_HOMES[i], 3+2+24+12+3, 16) for i in range(NUM_AGENTS)]
    world=World(agents,cap_A,cap_B,'no_crisis')
    for _ in range(TRAIN_EPISODES):
        world.reset_resources()
        for i,a in enumerate(agents):
            a.energy=15.; a.inventory=np.array(INIT_INV[i],dtype=float)
            a.signal[:]=0.; a.memory.fill(0); a.pos=CLOSE_HOMES[i]
        world.gift_log=[]; world.pending_gifts=[]; world.flipped=False
        tr=[]
        for s in range(TRAIN_STEPS):                 # 500 -- matches the eval horizon
            world.step(s, train=True, transitions=tr, agent_class='dqn')
        for g in world.gift_log:
            if not g['needed']: continue
            k=g['transition_index']
            if not (0 <= k < len(tr)): continue
            for t in tr[k+1:]:
                if t['agent_idx']!=g['receiver']: continue
                if t['step']-g['step']>10: break
                if t['action']==7 and t['reward']>=CONSUME_GAIN:
                    tr[k]['reward'] += GIVER_CREDIT_REWARD; break
        for t in tr:
            agents[t['agent_idx']].remember(t['obs'],t['action'],t['reward'],t['next_obs'],t['done'])
        for _ in range(25):
            for a in agents: a.replay()
    return agents


def evaluate(agents, condition, agent_class, seed):
    np.random.seed(seed+7777); random.seed(seed+7777); torch.manual_seed(seed+7777)
    cap_A,cap_B=build_capacity_maps()
    world=World(agents,cap_A,cap_B,condition)
    for i,a in enumerate(agents):
        a.energy=15.; a.inventory=np.array(INIT_INV[i],dtype=float)
        a.signal[:]=0.; a.memory.fill(0); a.pos=CLOSE_HOMES[i]
    acc={w: dict(surv=0, give=0, app=0, true=0, false=0, surp=0, miss=0,
                 pairs=[], strict=0) for w in WINDOWS}
    bounds={'pre':W_PRE, 'post1':W_POST1, 'late':W_LATE}
    for s in range(EVAL_STEPS):
        ev=world.step(s, train=False, track=True, agent_class=agent_class)
        for w,(lo,hi) in bounds.items():
            if lo <= s < hi:
                a_=acc[w]
                a_['surv'] += sum(ag.energy>0 for ag in agents)
                a_['give'] += ev['give_success']; a_['app'] += ev['app_inf']
                a_['true'] += ev['true_inf'];     a_['false'] += ev['false_cert']
                a_['surp'] += ev['surp_need'];    a_['miss']  += ev['miss_cert']
                a_['pairs'].extend(ev['give_pairs']); a_['strict'] += len(ev['cons_strict'])
    out={}
    for w,(lo,hi) in bounds.items():
        a_=acc[w]; n=hi-lo; g=max(a_['give'],1)
        m=np.zeros((3,3),int)
        for gi,ri in a_['pairs']: m[gi,ri]+=1
        out[f'survival_{w}']              = a_['surv']/(3*n)*100
        out[f'coop_{w}']                  = a_['give']
        out[f'apparent_informed_{w}']     = a_['app']/g
        out[f'true_informed_{w}']         = a_['true']/g
        out[f'false_certified_{w}']       = a_['false']/g
        out[f'certification_error_{w}']   = a_['false']/max(a_['app'],1)
        out[f'uncertified_true_need_{w}'] = a_['miss']/a_['surp'] if a_['surp']>0 else 0.
        out[f'strict_usefulness_{w}']     = a_['strict']/g
        out[f'reciprocity_{w}']           = float(np.any((m>0)&(m.T>0)&(np.eye(3)==0)))
    return out


def run_seed(seed):
    torch.set_num_threads(1)
    print(f'  [seed {seed}] training...', flush=True)
    trained = train_dqn(seed)
    rows=[]
    for cond in CONDITIONS:
        for kind in ('dqn','scripted'):
            if kind=='dqn':
                ags=[DQNAgent(i,i,CLOSE_HOMES[i], 3+2+24+12+3, 16) for i in range(NUM_AGENTS)]
                for i,a in enumerate(ags):
                    a.q_network.load_state_dict(trained[i].q_network.state_dict())
                    a.epsilon = 0.0
            else:
                ags=[ScriptedAgent(i,i,CLOSE_HOMES[i]) for i in range(NUM_AGENTS)]
            r = evaluate(ags, cond, kind, seed)
            r.update(condition=cond, agent_type=kind, seed=seed)
            rows.append(r)
    print(f'  [seed {seed}] done', flush=True)
    return rows

# ======================================================================
def analyse(path):
    import pandas as pd
    d = pd.read_csv(path)
    L=[]
    def say(s): print(s); L.append(s)
    def iqr(x):
        x=np.asarray(x,float)
        return f'{np.median(x):.3f} [{np.percentile(x,25):.3f}, {np.percentile(x,75):.3f}]'

    q = d[d.agent_type=='dqn']
    def get(cond, col): return q[q.condition==cond].sort_values('seed')[col].values
    def rate(cond, col, w): return get(cond, f'{col}_{w}') / WIN_LEN[w]

    say('\n================ REGISTERED OUTCOMES (DQN branch, v2) ================\n')

    # ---- STATIONARITY GATE ----
    surv_pre  = get('no_crisis','survival_pre');  surv_late = get('no_crisis','survival_late')
    coop_pre  = rate('no_crisis','coop','pre');   coop_late = rate('no_crisis','coop','late')
    ti_late   = get('no_crisis','true_informed_late')
    ok_s = surv_late >= GATE_SURV*surv_pre
    ok_c = coop_late >= GATE_COOP*coop_pre
    ok_t = ti_late   >= GATE_BAR
    gate = ok_s & ok_c & ok_t
    say('GATE -- is the no-crisis baseline STATIONARY over the evaluation horizon?')
    say(f'  survival        pre {iqr(surv_pre)} -> late {iqr(surv_late)}   ({int(ok_s.sum())}/{N_SEEDS} hold >= {GATE_SURV}x)')
    say(f'  cooperation/step pre {iqr(coop_pre)} -> late {iqr(coop_late)}  ({int(ok_c.sum())}/{N_SEEDS} hold >= {GATE_COOP}x)')
    say(f'  true-informed (late) {iqr(ti_late)}                            ({int(ok_t.sum())}/{N_SEEDS} clear {GATE_BAR})')
    say(f'  ALL THREE: {int(gate.sum())}/{N_SEEDS} (need {PASS_BAR})')
    if gate.sum() < PASS_BAR:
        say('\n  GATE FAILED.  The baseline is not stationary, so nothing can be attributed to')
        say('  the crisis.  C1-C4 WITHHELD.  This is the reported result.')
        open(f'{OUT_DIR}/xxii_summary.txt','w').write('\n'.join(L)); return
    say('  GATE PASSED -- proceeding to C1-C4.\n')

    # ---- C1 ----
    u  = get('cert_crisis_used_channel','uncertified_true_need_post1')
    od = get('ordinary_disturbance','uncertified_true_need_post1')
    nc = get('no_crisis','uncertified_true_need_post1')
    ok1 = ((u-od) >= C1_MARGIN) & ((od-nc) < C1_MARGIN)
    say('C1 -- certification crisis != ordinary disturbance')
    say(f'  uncertified-true-need (post1): no_crisis {iqr(nc)} | ordinary {iqr(od)} | crisis {iqr(u)}')
    say(f'  seeds passing: {int(ok1.sum())}/{N_SEEDS} (bar {PASS_BAR}) -> '
        f'{"PASS" if ok1.sum()>=PASS_BAR else "FAIL (null: the failure is generic, not certification-specific)"}')

    # ---- C2 ----
    d_miss  = u - nc
    d_false = get('cert_crisis_used_channel','false_certified_post1') - get('no_crisis','false_certified_post1')
    ok2 = d_miss > d_false
    say('\nC2 -- the signature is MISSED certification, not FALSE certification')
    say(f'  rise in uncertified-true-need : {iqr(d_miss)}')
    say(f'  rise in false-certified-give  : {iqr(d_false)}')
    say(f'  seeds passing: {int(ok2.sum())}/{N_SEEDS} (bar {C2_BAR}) -> '
        f'{"PASS: the system did not act on false certification; it lost the ability to certify real need" if ok2.sum()>=C2_BAR else "FAIL (null: symmetric, or false-certification-dominant)"}')
    say('  NOTE: C2 was the pilots\' most striking finding, but the pilots ran on a')
    say('  collapsing population.  It is re-registered here, not inherited, and it may not survive.')

    # ---- C3 ----
    say('\nC3 -- is there a recovery window?')
    delays, vals = [], []
    for dd in RESET_DELAYS:
        v = get(f'reset_d{dd}','true_informed_late')
        say(f'  reset at +{dd:3d}: late true-informed {iqr(v)}')
        delays += [dd]*len(v); vals += list(v)
    no_reset = get('cert_crisis_used_channel','true_informed_late')
    say(f'  no reset    : late true-informed {iqr(no_reset)}')
    rho, _ = spearmanr(delays, vals)
    diff = abs(float(np.median(get('reset_d100','true_informed_late')) - np.median(no_reset)))
    ok3a, ok3b = rho <= C3_RHO, diff < C3_NULLDIFF
    say(f'  Spearman rho(delay, recovery) = {rho:.3f} (bar <= {C3_RHO}) -> {"PASS" if ok3a else "FAIL"}')
    say(f'  |reset@100 - no reset| = {diff:.3f} (bar < {C3_NULLDIFF}) -> {"PASS" if ok3b else "FAIL"}')
    say(f'  C3 -> {"PASS: repairing the channel is not the same as restoring the coordination -- the difference is timing" if (ok3a and ok3b) else "FAIL (null: reset delay has no effect)"}')

    # ---- C4 ----
    cols=['true_informed_late','certification_error_late','uncertified_true_need_late']
    unused = np.stack([get('cert_crisis_unused_channel',c) for c in cols])
    ncl    = np.stack([get('no_crisis',c) for c in cols])
    ok4 = (np.abs(unused-ncl) < C4_TOL).all(axis=0)
    say('\nC4 -- specificity: a crisis on an UNUSED channel should do nothing')
    for k,c in enumerate(cols):
        say(f'  {c:28s}: no_crisis {iqr(ncl[k])} | unused-channel crisis {iqr(unused[k])}')
    say(f'  seeds passing: {int(ok4.sum())}/{N_SEEDS} (bar {PASS_BAR}) -> '
        f'{"PASS: a certification crisis only matters if it corrupts a channel the system actually uses" if ok4.sum()>=PASS_BAR else "FAIL (null: any signal corruption damages coordination -- which weakens C1)"}')

    say('\nScripted branch is in the CSV as a robustness check, not a registered outcome:')
    say('a fixed policy cannot recover by construction.\n')
    open(f'{OUT_DIR}/xxii_summary.txt','w').write('\n'.join(L))


def main():
    path=f'{OUT_DIR}/certification_crisis.csv'
    with Pool(processes=min(8, os.cpu_count() or 8)) as pool:
        all_rows = pool.map(run_seed, range(N_SEEDS))
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=CSV_FIELDS); w.writeheader()
        for rows in all_rows:
            for r in rows: w.writerow({k: r[k] for k in CSV_FIELDS})
    print(f'Wrote {path}')
    analyse(path)

if __name__ == '__main__':
    main()
