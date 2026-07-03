import numpy as np
import matplotlib.pyplot as plt

T = 1.0
N = 10000
dt = T / N

dW = np.random.normal(0, np.sqrt(dt), N)
W = np.concatenate(([0], np.cumsum(dW)))

empirical_qv = np.sum(dW**2)
total_variation = np.sum(np.abs(dW))

print(f"Theoretical Quadratic Variation: {T}")
print(f"Empirical Quadratic Variation:   {empirical_qv:.4f}")
print(f"Total Variation:                 {total_variation:.4f}")
print("Comment: The quadratic variation tightly converges near the theoretical value of T=1.")
print("Conversely, the total variation is massive. In theory, as N approaches infinity, total variation diverges to infinity, highlighting that Brownian paths are nowhere differentiable.\n")

N_values = [10, 50, 100, 500, 1000, 5000, 10000]
qv_results = []

for n_steps in N_values:
    dt_step = T / n_steps
    dW_step = np.random.normal(0, np.sqrt(dt_step), n_steps)
    qv = np.sum(dW_step**2)
    qv_results.append(qv)

plt.figure(figsize=(8, 5))
plt.plot(N_values, qv_results, marker='o', label='Empirical Quadratic Variation')
plt.axhline(y=T, color='r', linestyle='--', label='Theoretical T = 1')
plt.xscale('log')
plt.xlabel('Number of Partition Steps N (Log Scale)')
plt.ylabel('Quadratic Variation')
plt.title('Quadratic Variation vs Partition Steps')
plt.legend()
plt.grid(True, alpha=0.5)
plt.show()

print("Rate of convergence: As N increases (and the step size dt shrinks), the variance of the empirical QV drops.")
print("The plotted points converge progressively closer to the horizontal line T=1, visualizing the L2 convergence proven in part (d).\n")

t_fixed = 0.5
paths = 10000

W_t = np.random.normal(0, np.sqrt(t_fixed), paths)

empirical_mean = np.mean(W_t)
empirical_variance = np.var(W_t)

print(f"Theoretical E[W_t]:   0.0")
print(f"Empirical Mean:       {empirical_mean:.4f}")
print(f"Theoretical Var(W_t): {t_fixed}")
print(f"Empirical Variance:   {empirical_variance:.4f}")