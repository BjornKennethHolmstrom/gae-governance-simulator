import numpy as np
import matplotlib.pyplot as plt

def run_governance_simulation():
    # --- SIMULATION PARAMETERS ---
    T = 100  # Total time steps
    t_crisis = 20  # Time step when the crisis hits
    crisis_magnitude = -40.0  # The physical impact of the crisis
    equilibrium = 100.0  # Baseline system stability
    
    # System Dynamics (A and B matrices)
    # x(t+1) = A*x(t) + B*u(t-tau) + d(t)
    A_sys = 0.95  # Natural degradation without intervention (entropy)
    B_sys = 1.0   # Control input effectiveness

    # --- ARCHITECTURE A: TIER 1 (Centralized/Monoculture) ---
    # Fails the Law of Observability and the Separation Principle
    latency_A = 15          # Dead-time: It takes 15 steps for central node to act
    signal_noise_A = 8.0    # High noise: Bureaucratic distortion & hidden externalities
    gain_A = 0.3            # Weak control gain: Hard to tune due to high latency (Goodhart's trap)
    
    # --- ARCHITECTURE B: TIER 2 (Fractal/GGF) ---
    # Implements Subsidiarity, Hearts/Leaves Observability, and Separation Principle
    latency_B = 2           # Dead-time: Local BAZ reacts almost immediately
    signal_noise_B = 0.5    # Low noise: Dual-currency makes externalities visible
    gain_B = 0.8            # Strong control gain: Can be tuned aggressively due to low latency
    
    # Initialize state arrays
    x_A = np.full(T, equilibrium) # True Stability A
    x_B = np.full(T, equilibrium) # True Stability B
    
    y_A = np.zeros(T) # Observed State A
    y_B = np.zeros(T) # Observed State B
    
    u_A = np.zeros(T) # Control Action A
    u_B = np.zeros(T) # Control Action B

    # --- RUN SIMULATION ---
    for t in range(1, T-1):
        # 1. Disturbance (The Crisis)
        d_t = crisis_magnitude if t == t_crisis else 0
        
        # 2. Observation (Separation Principle & Signal Fidelity)
        # Architecture A measures with high distortion. B measures accurately.
        y_A[t] = x_A[t] + np.random.normal(0, signal_noise_A)
        y_B[t] = x_B[t] + np.random.normal(0, signal_noise_B)
        
        # 3. Control Law (Feedback Loop)
        # u(t) = K * Error. Controller calculates intervention based on what it observes.
        error_A = equilibrium - y_A[t]
        error_B = equilibrium - y_B[t]
        
        u_A[t] = gain_A * error_A
        u_B[t] = gain_B * error_B
        
        # 4. State Update (Latency/Dead-time integration)
        # Apply the control action from 'tau' steps ago
        action_A = u_A[t - latency_A] if t >= latency_A else 0
        action_B = u_B[t - latency_B] if t >= latency_B else 0
        
        # Calculate next state
        x_A[t+1] = A_sys * x_A[t] + B_sys * action_A + d_t + (equilibrium * (1 - A_sys))
        x_B[t+1] = A_sys * x_B[t] + B_sys * action_B + d_t + (equilibrium * (1 - A_sys))

    # --- PLOTTING THE PROOF ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot 1: True System Stability
    ax1.plot(range(T), x_A, label='Architecture A (Centralized/Tier 1)', color='#dc2626', linewidth=2)
    ax1.plot(range(T), x_B, label='Architecture B (Fractal GGF/Tier 2)', color='#16a34a', linewidth=2)
    ax1.axhline(equilibrium, color='gray', linestyle='--', alpha=0.5, label='Equilibrium')
    ax1.axvline(t_crisis, color='black', linestyle=':', label='Crisis Trigger')
    ax1.set_title('System Stability over Time (The Structural Proof)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True State (Stability)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Control Effort (Intervention Cost)
    ax2.plot(range(T), u_A, label='Arch A Intervention (Delayed, Chaotic)', color='#f87171', alpha=0.8)
    ax2.plot(range(T), u_B, label='Arch B Intervention (Rapid, Precise)', color='#4ade80', alpha=0.8)
    ax2.set_title('Control Effort / Governance Cost', fontsize=12)
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Intervention Magnitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_governance_simulation()
