import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

n = 10000
n_values = [10, 50, 200]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for i, n in enumerate(n_values):
    Sn = np.random.binomial(n, 0.5, n)
    Zn=(Sn - n/2)/np.sqrt(n/4)
    
    emp_mean = np.mean(Zn)
    emp_var = np.var(Zn)
    print(f"For n = {n:3}: Empirical Mean = {emp_mean:.4f}, Empirical Variance = {emp_var:.4f}")
    
    axes[i].hist(Zn, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black', label=f'Zn Histogram')

    x = np.linspace(-4, 4, 100)
    y = norm.pdf(x,0,1)
    axes[i].plot(x, y, 'r-', lw=2, label='Standard Normal PDF')
    
    axes[i].set_title(f'n = {n}')
    axes[i].set_xlabel('Zn')
    axes[i].set_xlim([-4, 4])
    axes[i].legend()

axes[0].set_ylabel('Density')
plt.suptitle('Empirical Central Limit Theorem Demonstration')
plt.tight_layout()
plt.show()