#!/usr/bin/env python3
"""
paper_xxii-certification_crisis_v4.py
=====================================
FINAL ATTEMPT AT THE ADAPTIVE BASELINE (Option B).  Stopping rule below.

WHY v4
------
v3 was WORSE than v2, and worse in training, not in evaluation: survival_pre fell from
100% (v2) to 45.7% (v3), cooperation from 0.307/step to 0.005/step.  The adaptive
evaluation was not the problem.  The trained policy was already broken.

THE BUG: epsilon decay was coupled to replay FREQUENCY.  `replay()` decays epsilon on
every call.  v2 called it 25x per episode; v3 called it every 8 steps of a 600-step
episode, i.e. 75x.  So epsilon reached its floor of 0.02 after ~1,300 replay calls --
about SEVENTEEN of six hundred episodes.  The learner stopped exploring almost at once
and never found trade.  The exploration schedule was never a schedule; it was a side
effect of an unrelated hyperparameter, and raising replay frequency silently cut it by
two thirds.

Compounding it: v3's consume-reward fix is correct but makes the learning signal
sparser and state-dependent (a flat +12 is a beacon; min(12, E_MAX - E) is not).
Correct and harder -- survivable with exploration, fatal without it.

WHAT v4 CHANGES, AND WHAT IT REFUSES TO CHANGE
----------------------------------------------
FIXED (bug):
  * Epsilon is now an explicit SCHEDULE over episodes, decoupled from replay entirely.
    Linear 1.0 -> 0.05 over the first EPS_DECAY_FRAC of training.  Changing replay
    frequency can no longer change how long the learner explores.

ADDED (principled, not tuned):
  * SURVIVAL_BONUS: a small per-step reward for being alive.  The framework's objective
    IS viability -- keeping essential variables inside V -- and a terminal death penalty
    encodes that only sparsely.  This is the dense form of the same objective, not a
    thumb on the scale for cooperation.  Nothing here rewards giving.

  * TRAIN_EPISODES 600 -> 1000.  Compute, not tuning.

NOT TOUCHED, deliberately: giver-credit magnitude, learning rate, death-penalty
magnitude, network size, gamma, the crisis manipulations, and every registered
threshold.  Those are the knobs that would turn this into a search for the result.

STOPPING RULE (committed before the run)
----------------------------------------
If the gate fails again, we STOP.  Two registered learner failures are reported as
results, and the demonstration falls back to Option A: the scripted branch, which
passes the stationarity gate at 17/20, with C3 honestly narrowed to STATE hysteresis
and Section 3.5 amended to say that policy hysteresis was not testable.

A fourth attempt would be tuning until the baseline appears, and at that point the
preregistration is decoration.  The learner failing is itself informative: it is the
no-trade equilibrium of exploration 19 -- coordination as a conditional attractor --
and it belongs to the multi-agent series, not to this paper.

Staged:  python3 paper_xxii-certification_crisis_v4.py gate     # no_crisis only
         python3 paper_xxii-certification_crisis_v4.py full     # only if gate passes
"""

import os, sys, csv, random, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from multiprocessing import Pool
from scipy.stats import spearmanr

# ======================================================================
# REGISTERED CONSTANTS  (unchanged from v3)
# ======================================================================
N_SEEDS      = 20
PASS_BAR     = 16
GATE_BAR     = 0.60
GATE_SURV    = 0.90
GATE_COOP    = 0.70
C1_MARGIN    = 0.10
C2_BAR       = 15
C3_RHO       = -0.50
C3_NULLDIFF  = 0.10
C4_TOL       = 0.10
RESET_DELAYS = [10, 25, 50, 100]

TRAIN_EPISODES = 1000      # was 600 -- compute, not tuning
TRAIN_STEPS    = 600
EVAL_STEPS     = 600
CRISIS_STEP    = 200
W_PRE   = (0,   200)
W_POST1 = (200, 250)
W_LATE  = (500, 600)
WIN_LEN = {'pre': 200, 'post1': 50, 'late': 100}

# --- exploration: an explicit schedule, decoupled from replay ---
EPS_START      = 1.00
EPS_END        = 0.05
EPS_DECAY_FRAC = 0.70      # linear decay over the first 70% of episodes
EVAL_EPS       = 0.05      # fixed at evaluation: it must be able to re-explore a repaired channel

CREDIT_WINDOW      = 10
TRAIN_REPLAY_EVERY = 4
EVAL_REPLAY_EVERY  = 1

OUT_DIR = 'xxii_out'; os.makedirs(OUT_DIR, exist_ok=True)

# ---------- World ----------
GRID_SIZE=5; NUM_AGENTS=3; RESOURCE_CAP=3; REGROWTH_RATE=0.12
ENERGY_MAX=20.0; INVENTORY_CAP=3; CONSUME_GAIN=12.0
GIVE_COST=0.3; METABOLIC_COST=0.4; GIVER_CREDIT_REWARD=2.0
DEATH_PENALTY=20.0
SURVIVAL_BONUS=0.1          # NEW: the dense form of "stay inside V".  Rewards nothing else.
HARVEST_MIN=0.5; HARVEST_FAIL_COST=0.05
GAMMA=0.99
HARVEST_EFFICIENCY = np.array([[2.,0.],[0.,2.],[1.2,1.2]])
CLOSE_HOMES=[(1,1),(1,3),(2,2)]
DIRS=[(-1,0),(1,0),(0,-1),(0,1)]
INIT_INV=[[2,0],[0,2],[1,1]]

CONDITIONS = (['no_crisis', 'ordinary_disturbance',
               'cert_crisis_used_channel', 'cert_crisis_unused_channel']
              + [f'reset_d{d}' for d in RESET_DELAYS])

METRICS = ['survival','coop','apparent_informed','true_informed','certified_give_rate',
           'false_certified','certification_error','uncertified_true_need',
           'strict_usefulness','reciprocity']
WINDOWS = ['pre','post1','late']
CSV_FIELDS = ['condition','agent_type','seed','all_survived',
              'surv_A','surv_B','surv_generalist'] + \
             [f'{m}_{w}' for m in METRICS for w in WINDOWS]

# ======================================================================
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

class DQN(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(i,128), nn.ReLU(),
                               nn.Linear(128,128), nn.ReLU(), nn.Linear(128,o))
    def forward(self,x): return self.net(x)

class DQNAgent:
    def __init__(self, idx, atype, pos, in_dim, act_dim):
        self.idx=idx; self.type=atype; self.pos=pos
        self.energy=15.; self.inventory=np.array(INIT_INV[idx],dtype=float)
        self.signal=np.array([0.,0.]); self.memory=np.zeros((NUM_AGENTS,6))
        self.act_dim=act_dim
        self.q_network=DQN(in_dim,act_dim); self.target_network=DQN(in_dim,act_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer=optim.Adam(self.q_network.parameters(), lr=1e-3)
        self.replay_buffer=deque(maxlen=20000)
        self.epsilon=EPS_START
        self.batch_size=64; self.update_target_every=200; self.step_count=0
    def act(self, obs):
        if random.random() < self.epsilon: return random.randrange(self.act_dim)
        with torch.no_grad():
            return int(torch.argmax(self.q_network(torch.FloatTensor(obs).unsqueeze(0))).item())
    def replay(self):
        """No epsilon decay here.  Exploration is a schedule, not a side effect of how
        often we happen to call this.  That coupling is what broke v3."""
        if len(self.replay_buffer) < self.batch_size: return
        b=random.sample(self.replay_buffer, self.batch_size)
        ob=torch.FloatTensor(np.array([t[0] for t in b]))
        ac=torch.LongTensor(np.array([t[1] for t in b])).unsqueeze(1)
        rw=torch.FloatTensor(np.array([t[2] for t in b]))
        nb=torch.FloatTensor(np.array([t[3] for t in b]))
        dn=torch.FloatTensor(np.array([t[4] for t in b]))
        q=self.q_network(ob).gather(1,ac).squeeze()
        with torch.no_grad():
            tgt = rw + GAMMA*self.target_network(nb).max(1)[0]*(1-dn)
        loss=nn.MSELoss()(q, tgt.detach())
        self.optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.)
        self.optimizer.step(); self.step_count+=1
        if self.step_count % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    def decay_memory(self, d=0.95): self.memory *= d

def epsilon_at(episode):
    frac = min(1.0, episode / max(1.0, EPS_DECAY_FRAC*TRAIN_EPISODES))
    return EPS_START + frac*(EPS_END - EPS_START)

class ScriptedAgent:
    def __init__(self, idx, atype, pos):
        self.idx=idx; self.type=atype; self.pos=pos
        self.energy=15.; self.inventory=np.array(INIT_INV[idx],dtype=float)
        self.signal=np.array([0.,0.]); self.memory=np.zeros((NUM_AGENTS,6))
    def decay_memory(self, d=0.95): self.memory *= d
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
            if w.resources[r,c,0]>=HARVEST_MIN and self.inventory[0]<INVENTORY_CAP: return 5
        elif self.type==1:
            if w.resources[r,c,1]>=HARVEST_MIN and self.inventory[1]<INVENTORY_CAP: return 6
        else:
            if self.inventory[0]<self.inventory[1] and w.resources[r,c,0]>=HARVEST_MIN and self.inventory[0]<INVENTORY_CAP: return 5
            if w.resources[r,c,1]>=HARVEST_MIN and self.inventory[1]<INVENTORY_CAP: return 6
            if w.resources[r,c,0]>=HARVEST_MIN and self.inventory[0]<INVENTORY_CAP: return 5
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
                        d=abs(r-o.pos[0])+abs(c-o.pos[1])
                        if d<best: best=d; tgt=o.pos
            if tgt is None: tgt=(2,2)
        valid=[d for d,(dr,dc) in enumerate(DIRS)
               if 0<=r+dr<GRID_SIZE and 0<=c+dc<GRID_SIZE and
               not any(o.energy>0 and o.pos==(r+dr,c+dc) for o in w.agents if o is not self)]
        if tgt and valid:
            return min(valid, key=lambda d: abs(r+DIRS[d][0]-tgt[0])+abs(c+DIRS[d][1]-tgt[1]))
        return random.choice(valid) if valid else 4

# ======================================================================
class World:
    def __init__(self, agents, cap_A, cap_B, condition='no_crisis'):
        self.agents=agents; self.cap_A=cap_A; self.cap_B=cap_B; self.condition=condition
        self.reset_step = (CRISIS_STEP + int(condition.split('_d')[1])
                           if condition.startswith('reset_d') else None)
        self.resources=np.zeros((GRID_SIZE,GRID_SIZE,2)); self.reset_resources()
        self.pending_gifts=[]; self.flipped=False; self.pending_tr=[]

    def reset_resources(self):
        self.resources[:,:,0]=np.random.rand(GRID_SIZE,GRID_SIZE)*self.cap_A
        self.resources[:,:,1]=np.random.rand(GRID_SIZE,GRID_SIZE)*self.cap_B

    def regrow(self, step):
        f = 0.5 if (self.condition=='ordinary_disturbance'
                    and CRISIS_STEP <= step < CRISIS_STEP+100) else 1.0
        self.resources[:,:,0]=np.minimum(self.resources[:,:,0]+REGROWTH_RATE*f, self.cap_A)
        self.resources[:,:,1]=np.minimum(self.resources[:,:,1]+REGROWTH_RATE*f, self.cap_B)

    def set_signals(self, step):
        crisis=(self.condition.startswith('cert_crisis') or self.condition.startswith('reset_d'))
        if crisis and step >= CRISIS_STEP: self.flipped=True
        if self.reset_step is not None and step >= self.reset_step: self.flipped=False
        for a in self.agents:
            if a.energy<=0: a.signal[:]=0.; continue
            if self.flipped and a.idx==1:
                if self.condition=='cert_crisis_unused_channel':
                    a.signal[0]=1. if a.inventory[0]<1. else 0.
                    a.signal[1]=1. if a.inventory[1]>=2. else 0.
                else:
                    a.signal[0]=1. if a.inventory[0]>=1. else 0.
                    a.signal[1]=1. if a.inventory[1]<1. else 0.
            else:
                a.signal[0]=1. if a.inventory[0]<1. else 0.
                a.signal[1]=1. if a.inventory[1]<1. else 0.

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

    def opportunities(self):
        o=dict(surp_need=0, miss_cert=0, certified=0)
        for i,j in itertools.combinations(range(NUM_AGENTS),2):
            ai,aj=self.agents[i],self.agents[j]
            if ai.energy<=0 or aj.energy<=0: continue
            if abs(ai.pos[0]-aj.pos[0])+abs(ai.pos[1]-aj.pos[1])!=1: continue
            for giver,taker in ((ai,aj),(aj,ai)):
                if giver.energy < GIVE_COST: continue
                for res in (0,1):
                    if giver.inventory[res] <= 1.: continue
                    if taker.inventory[res] < 1.:
                        o['surp_need']+=1
                        if taker.signal[res] < 0.5: o['miss_cert']+=1
                    if taker.signal[res] > 0.5:
                        o['certified']+=1
        return o

    def step(self, t, learn=False, track=False, agent_class='dqn', replay_every=4):
        self.set_signals(t)
        ev=dict(give_success=0, app_inf=0, true_inf=0, false_cert=0,
                cert_give=0, give_pairs=[], cons_strict=0) if track else None
        if track: ev.update(self.opportunities())

        order=list(range(NUM_AGENTS)); random.shuffle(order)
        for i in order:
            a=self.agents[i]
            if a.energy<=0: continue
            if agent_class=='dqn':
                obs=self.get_observation(i); action=a.act(obs)
            else:
                obs=None; action=a.decide(self)
            reward=self._apply(i, action, t, ev)
            a.energy=max(0., a.energy-METABOLIC_COST)
            if a.energy<=0: reward -= DEATH_PENALTY
            else:           reward += SURVIVAL_BONUS      # dense encoding of "inside V"
            if agent_class=='dqn' and learn:
                self.pending_tr.append(dict(agent_idx=i, step=t, obs=obs, action=action,
                                            reward=reward, next_obs=self.get_observation(i),
                                            done=a.energy<=0))
        if agent_class=='dqn' and learn:
            keep=[]
            for tr in self.pending_tr:
                if t - tr['step'] >= CREDIT_WINDOW:
                    self.agents[tr['agent_idx']].replay_buffer.append(
                        (tr['obs'], tr['action'], tr['reward'], tr['next_obs'], float(tr['done'])))
                else:
                    keep.append(tr)
            self.pending_tr=keep
            if t % replay_every == 0:
                for a in self.agents: a.replay()

        for a in self.agents: a.decay_memory()
        self.regrow(t)
        return ev

    def _credit_giver(self, receiver_idx, t):
        """A gift that was NEEDED and that the receiver actually converted into energy
        within the window pays the giver, credited in place while the transition is
        still pending."""
        for g in list(self.pending_gifts):
            if g['receiver']!=receiver_idx or not g['needed']: continue
            if t - g['step'] > CREDIT_WINDOW: continue
            for tr in self.pending_tr:
                if tr['agent_idx']==g['giver'] and tr['step']==g['step']:
                    tr['reward'] += GIVER_CREDIT_REWARD
                    break
            self.pending_gifts.remove(g)

    def _give(self, a, tgt, res, t, ev):
        accepted=min(1., INVENTORY_CAP-tgt.inventory[res])
        if accepted<=0 or a.inventory[res]<1 or a.energy<GIVE_COST: return -0.1
        true_need = tgt.inventory[res] < 1.
        app_need  = tgt.signal[res] > 0.5
        a.inventory[res]-=1; tgt.inventory[res]+=accepted; a.energy-=GIVE_COST
        self.pending_gifts.append(dict(receiver=tgt.idx, resource=res, step=t,
                                       giver=a.idx, needed=true_need))
        if ev is not None:
            ev['give_success']+=1; ev['give_pairs'].append((a.idx, tgt.idx))
            if app_need: ev['app_inf']+=1; ev['cert_give']+=1
            if true_need: ev['true_inf']+=1
            if app_need and not true_need: ev['false_cert']+=1
        return -GIVE_COST

    def _apply(self, i, action, t, ev):
        a=self.agents[i]; r,c=a.pos; reward=0.
        if action<4:
            dr,dc=DIRS[action]; nr,nc=r+dr,c+dc
            if 0<=nr<GRID_SIZE and 0<=nc<GRID_SIZE and \
               not any(o.energy>0 and o.pos==(nr,nc) for o in self.agents if o is not a):
                a.pos=(nr,nc)
        elif action==4: pass
        elif action in (5,6):
            res=action-5; eff=HARVEST_EFFICIENCY[a.type,res]; avail=self.resources[r,c,res]
            if eff<=0: reward-=0.1
            elif avail < HARVEST_MIN: reward-=HARVEST_FAIL_COST
            elif a.inventory[res] < INVENTORY_CAP:
                h=min(avail,1.); self.resources[r,c,res]-=h
                add=min(h*eff, INVENTORY_CAP-a.inventory[res]); a.inventory[res]+=add
                reward+=add*0.1
        elif action==7:
            if a.inventory[0]>=1 and a.inventory[1]>=1:
                gain=min(CONSUME_GAIN, ENERGY_MAX-a.energy)
                a.inventory-=1; a.energy+=gain; reward+=gain
                if gain>0: self._credit_giver(i, t)
                if ev is not None:
                    ev['cons_strict'] += sum(1 for g in self.pending_gifts
                                             if g['receiver']==i and t-g['step']<=CREDIT_WINDOW
                                             and g['needed'])
                self.pending_gifts=[g for g in self.pending_gifts
                                    if g['receiver']!=i and t-g['step']<=CREDIT_WINDOW]
            else: reward-=0.5
        elif 8<=action<=15:
            res=0 if action<12 else 1
            d=action-8 if action<12 else action-12
            dr,dc=DIRS[d]; tgt=None
            for o in self.agents:
                if o is not a and o.energy>0 and o.pos==(r+dr,c+dc): tgt=o
            if tgt is not None: reward += self._give(a, tgt, res, t, ev)
        return reward

# ======================================================================
def fresh_agents(kind):
    if kind=='dqn':
        return [DQNAgent(i,i,CLOSE_HOMES[i], 3+2+24+12+3, 16) for i in range(NUM_AGENTS)]
    return [ScriptedAgent(i,i,CLOSE_HOMES[i]) for i in range(NUM_AGENTS)]

def reset_agents(agents):
    for i,a in enumerate(agents):
        a.energy=15.; a.inventory=np.array(INIT_INV[i],dtype=float)
        a.signal[:]=0.; a.memory.fill(0); a.pos=CLOSE_HOMES[i]

def train_dqn(seed):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    cA,cB=build_capacity_maps()
    agents=fresh_agents('dqn')
    world=World(agents,cA,cB,'no_crisis')
    for ep in range(TRAIN_EPISODES):
        eps = epsilon_at(ep)                     # SCHEDULE, not a side effect of replay
        for a in agents: a.epsilon = eps
        world.reset_resources(); reset_agents(agents)
        world.pending_gifts=[]; world.pending_tr=[]; world.flipped=False
        for s in range(TRAIN_STEPS):
            world.step(s, learn=True, track=False, agent_class='dqn',
                       replay_every=TRAIN_REPLAY_EVERY)
    return agents

def evaluate(agents, condition, agent_class, seed, adapt):
    np.random.seed(seed+7777); random.seed(seed+7777); torch.manual_seed(seed+7777)
    cA,cB=build_capacity_maps()
    world=World(agents,cA,cB,condition); reset_agents(agents)
    world.pending_gifts=[]; world.pending_tr=[]
    acc={w: dict(surv=0, give=0, app=0, true=0, false=0, surp=0, miss=0,
                 certified=0, cert_give=0, pairs=[], strict=0) for w in WINDOWS}
    bounds={'pre':W_PRE,'post1':W_POST1,'late':W_LATE}
    for s in range(EVAL_STEPS):
        ev=world.step(s, learn=(adapt and agent_class=='dqn'), track=True,
                      agent_class=agent_class, replay_every=EVAL_REPLAY_EVERY)
        for w,(lo,hi) in bounds.items():
            if lo<=s<hi:
                a_=acc[w]
                a_['surv']+=sum(g.energy>0 for g in agents)
                a_['give']+=ev['give_success']; a_['app']+=ev['app_inf']
                a_['true']+=ev['true_inf'];     a_['false']+=ev['false_cert']
                a_['surp']+=ev['surp_need'];    a_['miss']+=ev['miss_cert']
                a_['certified']+=ev['certified']; a_['cert_give']+=ev['cert_give']
                a_['pairs'].extend(ev['give_pairs']); a_['strict']+=ev['cons_strict']
    out={'all_survived': float(all(a.energy>0 for a in agents)),
         'surv_A': float(agents[0].energy>0),
         'surv_B': float(agents[1].energy>0),
         'surv_generalist': float(agents[2].energy>0)}
    for w,(lo,hi) in bounds.items():
        a_=acc[w]; n=hi-lo; g=max(a_['give'],1)
        m=np.zeros((3,3),int)
        for gi,ri in a_['pairs']: m[gi,ri]+=1
        out[f'survival_{w}']              = a_['surv']/(3*n)*100
        out[f'coop_{w}']                  = a_['give']
        out[f'apparent_informed_{w}']     = a_['app']/g
        out[f'true_informed_{w}']         = a_['true']/g
        out[f'certified_give_rate_{w}']   = a_['cert_give']/a_['certified'] if a_['certified']>0 else 0.
        out[f'false_certified_{w}']       = a_['false']/g
        out[f'certification_error_{w}']   = a_['false']/max(a_['app'],1)
        out[f'uncertified_true_need_{w}'] = a_['miss']/a_['surp'] if a_['surp']>0 else 0.
        out[f'strict_usefulness_{w}']     = a_['strict']/g
        out[f'reciprocity_{w}']           = float(np.any((m>0)&(m.T>0)&(np.eye(3)==0)))
    return out

def run_seed(args):
    seed, conditions = args
    torch.set_num_threads(1)
    print(f'  [seed {seed}] training...', flush=True)
    trained=train_dqn(seed)
    rows=[]
    for cond in conditions:
        for kind in ('dqn','scripted'):
            ags=fresh_agents(kind)
            if kind=='dqn':
                for i,a in enumerate(ags):
                    a.q_network.load_state_dict(trained[i].q_network.state_dict())
                    a.target_network.load_state_dict(trained[i].q_network.state_dict())
                    a.epsilon=EVAL_EPS
            r=evaluate(ags, cond, kind, seed, adapt=True)
            r.update(condition=cond, agent_type=kind, seed=seed)
            rows.append(r)
    print(f'  [seed {seed}] done', flush=True)
    return rows

# ======================================================================
def _iqr(x):
    x=np.asarray(x,float)
    if not len(x): return 'n/a'
    return f'{np.median(x):.3f} [{np.percentile(x,25):.3f}, {np.percentile(x,75):.3f}]'

def gate_report(d, say):
    q=d[(d.agent_type=='dqn')&(d.condition=='no_crisis')].sort_values('seed')
    sp,sl=q.survival_pre.values, q.survival_late.values
    cp,cl=q.coop_pre.values/WIN_LEN['pre'], q.coop_late.values/WIN_LEN['late']
    ti=q.true_informed_late.values
    ok_s, ok_c, ok_t = sl>=GATE_SURV*sp, cl>=GATE_COOP*cp, ti>=GATE_BAR
    gate=ok_s & ok_c & ok_t
    say('GATE -- is the ADAPTIVE no-crisis baseline stationary and cooperative?')
    say(f'  survival         pre {_iqr(sp)} -> late {_iqr(sl)}   ({int(ok_s.sum())}/{len(q)})')
    say(f'  cooperation/step pre {_iqr(cp)} -> late {_iqr(cl)}   ({int(ok_c.sum())}/{len(q)})')
    say(f'  true-informed (late) {_iqr(ti)}                      ({int(ok_t.sum())}/{len(q)})')
    say(f'  certified-give rate (late) {_iqr(q.certified_give_rate_late.values)}   [trust in the channel]')
    say('')
    say('  WHO SURVIVES?  (the no-trade equilibrium is the generalist surviving alone)')
    say(f'    A-specialist  : {q.surv_A.mean()*100:5.1f}% of seeds')
    say(f'    B-specialist  : {q.surv_B.mean()*100:5.1f}% of seeds')
    say(f'    generalist    : {q.surv_generalist.mean()*100:5.1f}% of seeds')
    say(f'    all three     : {q.all_survived.mean()*100:5.1f}% of seeds')
    say('')
    say(f'  ALL THREE GATE CRITERIA: {int(gate.sum())}/{len(q)} (need {PASS_BAR})')
    return int(gate.sum()) >= PASS_BAR

# (analyse_full identical in structure to v3: C1, C2, C3a state / C3b policy, C4)
def analyse_full(path):
    import pandas as pd
    d=pd.read_csv(path); L=[]
    def say(s): print(s); L.append(s)
    q=d[d.agent_type=='dqn']
    def get(cond,col): return q[q.condition==cond].sort_values('seed')[col].values
    say('\n========== REGISTERED OUTCOMES (adaptive DQN branch, v4) ==========\n')
    if not gate_report(d, say):
        say('\n  GATE FAILED (second time).  STOPPING RULE APPLIES.')
        say('  C1-C4 WITHHELD.  Two registered learner failures are the reported result.')
        say('  Fall back to Option A: the scripted branch (17/20 on the stationarity gate),')
        say('  with C3 narrowed to STATE hysteresis and Section 3.5 amended accordingly.')
        open(f'{OUT_DIR}/xxii_summary.txt','w').write('\n'.join(L)); return
    say('  GATE PASSED.\n')

    u  = get('cert_crisis_used_channel','uncertified_true_need_post1')
    od = get('ordinary_disturbance','uncertified_true_need_post1')
    nc = get('no_crisis','uncertified_true_need_post1')
    ok1=((u-od)>=C1_MARGIN)&((od-nc)<C1_MARGIN)
    say('C1 -- certification crisis != ordinary disturbance')
    say(f'  uncertified-true-need (post1): no_crisis {_iqr(nc)} | ordinary {_iqr(od)} | crisis {_iqr(u)}')
    say(f'  {int(ok1.sum())}/{N_SEEDS} (bar {PASS_BAR}) -> {"PASS" if ok1.sum()>=PASS_BAR else "FAIL"}')

    d_miss=u-nc
    d_false=get('cert_crisis_used_channel','false_certified_post1')-get('no_crisis','false_certified_post1')
    ok2=d_miss>d_false
    say('\nC2 -- the signature is MISSED certification, not FALSE certification')
    say(f'  rise in uncertified-true-need : {_iqr(d_miss)}')
    say(f'  rise in false-certified-give  : {_iqr(d_false)}')
    say(f'  {int(ok2.sum())}/{N_SEEDS} (bar {C2_BAR}) -> {"PASS" if ok2.sum()>=C2_BAR else "FAIL"}')

    say('\nC3a -- STATE hysteresis: does the population recover?')
    dd_,sv=[],[]
    for x in RESET_DELAYS:
        v=get(f'reset_d{x}','survival_late'); say(f'  reset +{x:3d}: survival(late) {_iqr(v)}')
        dd_+=[x]*len(v); sv+=list(v)
    say(f'  Spearman rho(delay, survival) = {spearmanr(dd_,sv)[0]:.3f}')

    say('\nC3b -- POLICY hysteresis, among seeds where EVERY agent survived')
    dl,tr=[],[]
    for x in RESET_DELAYS:
        sub=q[(q.condition==f'reset_d{x}')&(q.all_survived==1.0)]
        v=sub.certified_give_rate_late.values
        say(f'  reset +{x:3d}: certified-give(late) {_iqr(v)}   (n={len(v)})')
        dl+=[x]*len(v); tr+=list(v)
    nores=q[(q.condition=='cert_crisis_used_channel')&(q.all_survived==1.0)].certified_give_rate_late.values
    base =q[(q.condition=='no_crisis')&(q.all_survived==1.0)].certified_give_rate_late.values
    say(f'  no crisis {_iqr(base)} | no reset {_iqr(nores)}')
    if len(dl)>=8:
        rho=spearmanr(dl,tr)[0]
        d100=q[(q.condition=='reset_d100')&(q.all_survived==1.0)].certified_give_rate_late.values
        diff=abs(float(np.median(d100)-np.median(nores))) if len(d100) and len(nores) else np.nan
        okA, okB = rho<=C3_RHO, diff<C3_NULLDIFF
        say(f'  rho(delay, trust) = {rho:.3f} (bar <= {C3_RHO}) -> {"PASS" if okA else "FAIL"}')
        say(f'  |reset@100 - no reset| = {diff:.3f} (bar < {C3_NULLDIFF}) -> {"PASS" if okB else "FAIL"}')
        say(f'  C3b -> {"PASS: repairing the channel is not restoring the coordination -- the difference is TIMING" if (okA and okB) else "FAIL (null: a surviving controller always re-acquires trust)"}')
    else:
        say('  TOO FEW SURVIVING SEEDS -- inconclusive, reported as such, not folded into C3a.')

    cols=['true_informed_late','certification_error_late','uncertified_true_need_late']
    un=np.stack([get('cert_crisis_unused_channel',c) for c in cols])
    nl=np.stack([get('no_crisis',c) for c in cols])
    ok4=(np.abs(un-nl)<C4_TOL).all(axis=0)
    say('\nC4 -- specificity: a crisis on an UNUSED channel should do nothing')
    for k,c in enumerate(cols):
        say(f'  {c:28s}: no_crisis {_iqr(nl[k])} | unused {_iqr(un[k])}')
    say(f'  {int(ok4.sum())}/{N_SEEDS} (bar {PASS_BAR}) -> {"PASS" if ok4.sum()>=PASS_BAR else "FAIL"}')
    open(f'{OUT_DIR}/xxii_summary.txt','w').write('\n'.join(L))

# ======================================================================
def main():
    stage = sys.argv[1] if len(sys.argv)>1 else 'gate'
    conds = ['no_crisis'] if stage=='gate' else CONDITIONS
    path=f'{OUT_DIR}/certification_crisis_v4_{stage}.csv'
    with Pool(processes=min(8, os.cpu_count() or 8)) as pool:
        allr=pool.map(run_seed, [(s, conds) for s in range(N_SEEDS)])
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=CSV_FIELDS); w.writeheader()
        for rows in allr:
            for r in rows: w.writerow({k: r[k] for k in CSV_FIELDS})
    print(f'Wrote {path}\n')
    if stage=='gate':
        import pandas as pd
        L=[]
        def say(s): print(s); L.append(s)
        say('============ STAGE 1: BASELINE GATE (no_crisis only) ============\n')
        ok=gate_report(pd.read_csv(path), say)
        say('')
        say('  -> PROCEED to `full`.' if ok else
            '  -> STOP.  STOPPING RULE APPLIES: this is the second registered failure of the\n'
            '     adaptive learner.  Do not attempt a fourth configuration.  Report both\n'
            '     failures and fall back to Option A (scripted branch), narrowing C3 to\n'
            '     state hysteresis and amending Section 3.5.')
        open(f'{OUT_DIR}/xxii_gate.txt','w').write('\n'.join(L))
    else:
        analyse_full(path)

if __name__ == '__main__':
    main()
