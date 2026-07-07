# Preregistration — Goodhart intervention-set demo (Paper XX)

**Status:** Written before the run. There is no pilot; the explorations behind Paper XX (`08-goodhart.md`) are argument, not simulation. This demo is the paper's one empirical result and is registered from scratch.

## What the demo tests

The intervention-set theorem (Paper XX §4) states: a proxy $M$ is safe under optimization iff, over the *reachable intervention set*, its induced ordering refines the target ordering. The corollary this demo isolates, stated precisely: **reachability of the discarded target-relevant dimension is necessary for proxy optimization to create Goodhart degradation away from an initially target-optimal state.** When that dimension is frozen at its target-optimal value, the same lossy proxy is harmless. (The demo does not show the broader claim that a lossy proxy is safe whenever the dimension is unreachable *from an arbitrary state* — a dimension frozen at a bad value stays bad, and optimizing the measured dimension cannot repair it. The claim is specifically about degradation from the optimum.)

The demo tests this corollary. Demonstrating that Goodhart *can* happen is trivial; the informative content is the contrast — the *same* lossy proxy is safe at $r=0$ (discarded dimension frozen at its optimum) and unsafe at $r=1$ (fully reachable), with severity scaling monotonically in between. Note honestly that the $r=0$ safety is analytic, not an empirical null (see P2), so the load of the empirical test falls on P3: that degradation is *governed by reachability* rather than merely present.

## The world

Each seed is one world. A fixed budget $C$ is allocated between a *measured* dimension $a$ and a *hidden* dimension $b$, with $a + b \le C$. The target rewards both through a concave utility, $T = \alpha\,a^{p} + \beta\,b^{p}$ with $0<p<1$, $\alpha,\beta>0$. The proxy is the projection that drops the hidden dimension: $M = a^{p}$. Per seed, drawn i.i.d.: $C \sim U(5,20)$, $p \sim U(0.3,0.7)$, $\alpha,\beta \sim U(0.8,1.2)$. Because $\beta>0$ always, the hidden dimension is always target-relevant; the manipulated variable is whether it is *reachable*. (The weight range is deliberately narrow: wider asymmetry produces worlds where the hidden dimension, though target-relevant, is weak enough that fully starving it costs little, which weakens the exploit without illuminating the theorem.)

The true optimum allocates the budget to equalize marginal returns (closed form: $a^\*/b^\* = (\beta/\alpha)^{1/(p-1)}$, scaled to $a^\*+b^\*=C$), giving target $T^\*$. The optimizer instead maximizes the proxy $M=a^{p}$, i.e. drives $a$ up.

**Reachability fraction $r \in [0,1]$.** The hidden dimension may be reduced from $b^\*$ down to $(1-r)\,b^\*$; the freed budget goes to $a$. $r=0$: $b$ is frozen at $b^\*$ (unreachable) — the budget constraint then forces the proxy optimum onto $a^\*$, so $M$-optimization lands on $T^\*$. $r=1$: $b$ can be starved fully ($b\to 0$), the classic exploit. Degradation is $D(r) = (T^\* - T(r))/T^\*$.

## Registered predictions

**P1 — Goodhart under a reachable exploit.** At $r=1$, proxy optimization degrades the target: $D(1) > 0.10$ in $\ge 24/30$ seeds (80%).
*Null committed to:* if $D(1) \le 0.10$ in a majority, the exploit is too weak to constitute Goodhart at this parameterization and the demo is uninformative; report and diagnose.

**P2 — Analytic sanity check (safety at the frozen optimum).** At $r=0$ the hidden dimension is frozen at $b^\*$, so the budget constraint forces $a=a^\*$ and $D(0)=0$ by construction. This is not an empirical null — it is a check that the implementation reproduces the theorem's frozen-at-optimum case: $D(0) < 0.01$ in $\ge 28/30$ seeds (numerical slack only).
*If this fails* it signals a code error, not a substantive result.

**P3 — Severity scales with reachability.** Across $r \in \{0,0.25,0.5,0.75,1.0\}$, degradation is monotone non-decreasing in $r$: pooled Spearman$(r, D) > 0.9$, and per-seed monotonicity (no decrease between adjacent $r$) holds in $\ge 27/30$ seeds.
*Null committed to:* if the pooled Spearman $\le 0.9$, severity is not controlled by reachability and the quantitative form of the theorem is unsupported.

## Analysis plan

Closed-form per seed (no training, no stochastic optimization — the optimum of $M=a^p$ over each reachable set is analytic), so results are deterministic given the seed's world parameters; the only randomness is the world draw. Report $D$ at each $r$ as median and IQR across seeds, the three pass/fail counts, and one figure: mean $M$ and mean $T$ versus $r$ (the Goodhart scissors — proxy rising, target falling). No seed excluded except on a numerical failure (reported). No post-hoc threshold changes.

## What failure would mean

- P1 fails → the exploit is mis-parameterized; the demo does not show Goodhart and the §4 theorem keeps only its analytic support.
- **P2 fails → implementation error** (the frozen-at-optimum case should give $D(0)=0$ analytically); not a substantive result, but a signal to fix the code before trusting P1/P3.
- P3 fails → Goodhart is present but not governed by reachability; the theorem's *quantitative* claim (severity tracks the reachable set) is withdrawn. This is the empirically load-bearing prediction, since P2 is analytic and P1 only establishes the exploit exists.
