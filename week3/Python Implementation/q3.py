import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

X0 = 3.0
theta = 2.0
sigma = 0.5

T_d = 5.0
N_d = 5000
dt_d = T_d / N_d
t_d = np.linspace(0, T_d, N_d + 1)
stat_std = np.sqrt(sigma**2 / (2 * theta))

plt.figure(figsize=(10, 4))
for _ in range(10):
    X = np.zeros(N_d + 1)
    X[0] = X0
    for k in range(N_d):
        dW = np.random.normal(0, np.sqrt(dt_d))
        X[k+1] = X[k] - theta * X[k] * dt_d + sigma * dW
    plt.plot(t_d, X, lw=1)

plt.axhline(0, color='k', linestyle='--', label='Stationary Mean (0)')
plt.fill_between(t_d, -2*stat_std, 2*stat_std, color='gray', alpha=0.3, label='±2 Std Dev Band')
plt.title('(d) 10 Paths of OU Process')
plt.legend()
plt.show()

T_e = 10.0
N_e = int(T_e / dt_d) 
paths_e = 20000

X_e = np.full(paths_e, X0)
for _ in range(N_e):
    dW = np.random.normal(0, np.sqrt(dt_d), paths_e)
    X_e += -theta * X_e * dt_d + sigma * dW

emp_mean = np.mean(X_e)
emp_var = np.var(X_e)
theo_var = sigma**2 / (2 * theta)

print(f"(e) Empirical Mean: {emp_mean:.4f} (Theo: 0.0000)")
print(f"(e) Empirical Var:  {emp_var:.4f} (Theo: {theo_var:.4f})\n")

plt.figure(figsize=(6, 4))
plt.hist(X_e, bins=60, density=True, alpha=0.6, label='Empirical Terminal X')
x_axis = np.linspace(-1, 1, 100)
plt.plot(x_axis, norm.pdf(x_axis, 0, stat_std), 'r-', lw=2, label='Theoretical PDF')
plt.title('(e) Stationary Distribution')
plt.legend()
plt.show()

thetas = [0.5, 2.0, 5.0]
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

print("(f) Effects of Theta:")
print("- Speed of return: Higher theta pulls paths back to 0 much faster.")
print("- Width: Higher theta tightly constricts the stationary distribution (lower variance).\n")

for i, th in enumerate(thetas):
    for _ in range(5):
        X = np.zeros(N_d + 1)
        X[0] = X0
        for k in range(N_d):
            dW = np.random.normal(0, np.sqrt(dt_d))
            X[k+1] = X[k] - th * X[k] * dt_d + sigma * dW
        axes[i].plot(t_d, X, lw=1)
    axes[i].axhline(0, color='k', linestyle='--')
    axes[i].set_title(f'Theta = {th}')

plt.suptitle('(f) OU Process with Varying Theta')
plt.show()