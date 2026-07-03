import numpy as np
import matplotlib.pyplot as plt

N_values = [10**2, 10**3, 10**4, 10**5]
true_pi = np.pi

estimates = []
errors = []

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

np.random.seed(42)

for N in N_values:
    x = np.random.uniform(-1, 1, N)
    y = np.random.uniform(-1, 1, N)
    
    inside_circle = (x**2 + y**2) <= 1
    num_inside = np.sum(inside_circle)
    
    pi_estimate = 4 * (num_inside / N)
    estimates.append(pi_estimate)
    
    error = np.abs(pi_estimate - true_pi)
    errors.append(error)
    print(f"N = {N:<7}: Estimated Pi = {pi_estimate:.5f}, Error = {error:.5f}")

    if N == 10**4:
        axes[0].scatter(x[inside_circle], y[inside_circle], color='blue', s=2, label='Inside Circle')
        axes[0].scatter(x[~inside_circle], y[~inside_circle], color='red', s=2, label='Outside Circle')
        
        circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=2)
        axes[0].add_artist(circle)
        
        axes[0].set_aspect('equal', 'box')
        axes[0].set_xlim([-1, 1])
        axes[0].set_ylim([-1, 1])
        axes[0].set_title(f'Monte Carlo Simulation (N = {N})')
        axes[0].legend(loc='upper right')

axes[1].plot(N_values, estimates, marker='o', linestyle='-', color='purple', label='Estimated $\pi$')
axes[1].axhline(y=true_pi, color='green', linestyle='--', label='True $\pi$ (3.14159...)')
axes[1].set_xscale('log')
axes[1].set_xlabel('Number of Samples (N)')
axes[1].set_ylabel('Estimate of $\pi$')
axes[1].set_title('Convergence of $\pi$ Estimate')
axes[1].legend()
axes[1].grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.show()