"""
self_ii_appendix_e_observer_plant.py

Reproduces the numerical claims in Self II, Appendix E
(Observer-Plant Identity and the Measurement-Disturbance Coupling):
self-perception as fixed-point selection (E.2) and self-observation
as intervention / the relocation lever (E.3).

Flat repo: gae-governance-simulator/
Run: python self_ii_appendix_e_observer_plant.py
"""


def observe_self(b0, eta, bias, x0=0.5, lam=0.2, T=4000,
                 intervene_at=None, intervene_to=None):
    """Reflexive self-observation of one self-dimension.

    eta   observation-as-action: believing pulls the true state toward the belief
          (eta = 0 recovers a PASSIVE observer with a fixed external truth)
    bias  confirmation: the reading is drawn toward the current belief
    intervene_at / intervene_to : optional one-time deliberate belief shift (re-narration)

    Returns final (true state x, belief b).
    """
    x, b = x0, b0
    for t in range(T):
        if intervene_at is not None and t == intervene_at:
            b = intervene_to
        x = x + eta * (b - x)              # observing/believing moves the state (Part I 1.3/1.4)
        obs = x + bias * (b - x)           # reading drawn toward belief (confirmation)
        b = b + lam * (obs - b)            # belief updates toward the (biased) reading
    return round(x, 3), round(b, 3)


if __name__ == "__main__":
    print("E.2  Passive vs reflexive self-observation (common true initial state x0 = 0.5):")
    print("     passive converges to the truth 0.5; reflexive self-confirms wherever belief starts")
    for b0 in (0.1, 0.3, 0.5, 0.7, 0.9):
        p = observe_self(b0, eta=0.0, bias=0.0)
        r = observe_self(b0, eta=0.15, bias=0.5)
        print(f"     b0={b0}:  passive (x,b)={p}   reflexive (x,b)={r}")

    print("E.3  Self-observation as intervention: belief shift -> 0.65 at t=1500 (start locked at 0.2):")
    print("     reflexive:", observe_self(0.2, 0.15, 0.5, x0=0.2, intervene_at=1500, intervene_to=0.65),
          "-> relocates and holds (partway: set by eta vs bias)")
    print("     passive  :", observe_self(0.2, 0.0, 0.0, x0=0.2, intervene_at=1500, intervene_to=0.65),
          "-> returns to the fixed truth 0.2")
