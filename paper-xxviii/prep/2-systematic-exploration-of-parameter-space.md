## 2. Model Definition and Governance Interpretation

We model a single governance system as a coupled five‑variable dynamical system. The variables are all dimensionless, bounded in \([0,1]\), and represent aggregate properties of the system rather than specific institutional features. The model is intended as a minimal abstraction: it captures feedback mechanisms that recur across many governance contexts without claiming to reproduce any particular institution.

### 2.1 State variables

| Variable | Meaning | Governance interpretation |
|---|---|---|
| \(U\) | Actual unresolved environmental uncertainty | How much of the world's current state is not captured by the system's models and procedures. \(U=0\): fully legible; \(U=1\): wholly opaque. |
| \(B\) | Boundary strength / closure | The intensity of institutional separation: rules, borders, classification, surveillance, doctrinal commitment. \(B=0\): fully porous; \(B=1\): total closure. |
| \(T\) | Trust capacity | The ability to coordinate under unresolved uncertainty without requiring additional boundary closure. \(T=0\): no trust; \(T=1\): full trust. |
| \(E\) | Exploratory capacity | The system's ability to interact with its environment and reduce uncertainty through learning, experimentation, and engagement. \(E=0\): no exploration; \(E=1\): maximal exploration. |
| \(P\) | Boundary permeability | The degree to which information and adaptation can cross boundaries. \(P=0\): opaque, impermeable; \(P=1\): fully transparent and open to information flow. |

The key distinction introduced in this model is between **boundary strength** \(B\) and **boundary permeability** \(P\). A boundary may be strong but permeable—for example, a quarantine that blocks physical movement but allows data and expertise to flow—or strong and impermeable—for example, a totalitarian information blackout. The variable \(P\) captures this second dimension.

### 2.2 Auxiliary quantities

Define **felt uncertainty** \(F\) as

\[
F = \frac{s U}{(1+\lambda T)(1+\mu B)},
\]

where \(s\) is the stakes/uncertainty multiplier, and \(\lambda,\mu>0\) are constants. Felt uncertainty is the system's internal experience of uncertainty. It is reduced both by trust \(T\) and by boundary strength \(B\), even if these do not reduce actual uncertainty \(U\). This captures the familiar phenomenon that governments often build walls to make the world *feel* more predictable, rather than to make it more predictable.

Define the **effective boundary suppression block** \(Q\) as

\[
Q = (1-P)B.
\]

This is the portion of boundary strength that actually suppresses trust and exploration. If permeability is high (\(P \approx 1\)), even a strong boundary does not damage the system's adaptive capacities. If permeability is low, the boundary becomes harmful.

### 2.3 Dynamical equations

The time evolution is given by:

\[
\begin{aligned}
\dot U &= n(1-U) - \alpha E (1-\beta Q) U, \\
\dot B &= \rho_B \, \sigma\!\big(k_B(F-\theta)\big) - d_B B, \\
\dot T &= \rho_T E (1-\beta_T Q) - d_T T - \gamma Q T, \\
\dot E &= \rho_E \, \sigma\!\Big( k_E \big( \frac{\alpha U}{1+\eta Q} - c_E \big) \Big) - d_E E, \\
\dot P &= \rho_P \left( 1 - \sigma\!\big( k_P(F-\theta_P) \big) - P \right),
\end{aligned}
\]

where \(\sigma(z) = 1/(1+e^{-z})\) is the logistic sigmoid, and all parameters are positive.

Each equation has a direct governance interpretation.

**Uncertainty equation.** The term \(n(1-U)\) represents the natural inflow of new disturbances, novelties, and environmental changes. The second term represents reduction of uncertainty through effective exploration. Boundaries reduce exploration's effectiveness if they are strong and opaque, via the factor \((1-\beta Q)\). Thus a closed system may be unable to reduce its actual uncertainty even if it feels secure.

**Boundary equation.** Boundary investment is driven by felt uncertainty: when \(F\) exceeds a tolerance threshold \(\theta\), the system builds boundaries. The sigmoid function \(\sigma(k_B(F-\theta))\) is a smooth switch. Boundaries decay at rate \(d_B\), representing institutional inertia and automatic sunset. The parameter \(k_B\) controls how sharply the system responds to felt uncertainty.

**Trust equation.** Trust grows through exploration under open conditions, but only if the boundary is not too opaque. The term \(\rho_T E (1-\beta_T Q)\) captures this: exploration builds trust when it is not blocked. Trust decays at rate \(d_T\), and is further eroded by the boundary suppression block \(Q\) through the term \(-\gamma Q T\). Thus an opaque boundary actively destroys trust.

**Exploration equation.** Exploration is triggered when actual uncertainty is high relative to a cost threshold \(c_E\), but is suppressed by opaque boundaries. The sigmoid \(\sigma(k_E(\alpha U/(1+\eta Q)-c_E))\) captures the decision to explore: if the expected information gain \(\alpha U\) exceeds the cost, exploration is active. Opaque boundaries raise the effective cost of exploration by a factor \(1+\eta Q\). Exploration also decays at rate \(d_E\), representing fatigue, budget cycles, or institutional forgetting.

**Permeability equation.** Permeability adapts slowly, with rate \(\rho_P\), which is much smaller than the other rates. It tends toward a target value \(1-\sigma(k_P(F-\theta_P))\): when felt uncertainty is low, the target is near 1 (openness); when felt uncertainty is high, the target is near 0 (opacity). The threshold \(\theta_P\) is distinct from \(\theta\), allowing the quality of boundaries to degrade before the system necessarily builds more of them. This slow equation is the model's representation of institutional trust erosion, media freedom, legal protections, and other slow-moving governance features.

### 2.4 Parameters and default values

For reproducibility, the parameter values used in the simulations are:

| Parameter | Default | Description |
|---|---:|---|
| \(n\) | 0.120 | Natural rate of uncertainty inflow |
| \(\alpha\) | 1.339 | Exploration efficiency in reducing uncertainty |
| \(\beta\) | 0.539 | Boundary suppression effect on uncertainty reduction |
| \(s\) | 0.908 | Base stakes multiplier |
| \(\lambda\) | 2.700 | Trust's reduction of felt uncertainty |
| \(\mu\) | 1.841 | Boundary's reduction of felt uncertainty |
| \(\theta\) | 0.196 | Felt uncertainty threshold for boundary investment |
| \(k_B\) | 23.591 | Sharpness of boundary response |
| \(\rho_B\) | 0.154 | Boundary growth rate |
| \(d_B\) | 0.117 | Boundary decay rate |
| \(\rho_T\) | 0.546 | Trust growth rate |
| \(\beta_T\) | 0.766 | Boundary suppression effect on trust growth |
| \(d_T\) | 0.067 | Trust decay rate |
| \(\gamma\) | 0.110 | Boundary suppression effect on trust maintenance |
| \(\rho_E\) | 0.073 | Exploration growth rate |
| \(\eta\) | 2.065 | Boundary suppression effect on exploration drive |
| \(c_E\) | 0.489 | Exploration cost threshold |
| \(k_E\) | 24.382 | Sharpness of exploration response |
| \(d_E\) | 0.059 | Exploration decay rate |
| \(\rho_P\) | 0.02 | Permeability adaptation rate (slow) |
| \(k_P\) | 20.0 | Sharpness of permeability response |
| \(\theta_P\) | 0.15 | Felt uncertainty threshold for permeability loss |

The time unit is arbitrary. Rates can be rescaled without changing the qualitative behaviour, provided the separation of timescales between \(\rho_P\) and the other rates is preserved.

### 2.5 Scope and limitations

The model is not a calibrated empirical model. Its purpose is to expose qualitative mechanisms that are analytically and computationally tractable, and to generate testable hypotheses for more detailed institutional studies. The variables are aggregate constructs that condense many institutional features into single numbers, and the thresholds and sigmoids are deliberately sharp approximations to the continuous, heterogeneous processes in real governance systems. All numerical results should be read as **[R within model]**, and any translation to real institutions as **[IP]**.

With the model defined, Section 3 turns to the analytical structure of the fast–slow dynamics and the conditions under which bistability arises.
