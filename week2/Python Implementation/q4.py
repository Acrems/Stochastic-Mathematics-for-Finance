import numpy as np
import matplotlib.pyplot as plt

S0 = 100
K = 110
num_simulations = 100000
np.random.seed(42)

Z = np.random.normal(0, 1, num_simulations)

sigma_base = 0.2
ST_base = S0 * np.exp(sigma_base * Z)

payoffs_base = np.maximum(ST_base - K, 0)

expected_payoff_base = np.mean(payoffs_base)
print(f"Estimated Expected Payoff (sigma = {sigma_base}): {expected_payoff_base:.4f}")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(ST_base, bins=60, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].axvline(K, color='red', linestyle='dashed', linewidth=2, label=f'Strike (K={K})')
axes[0].set_title('Simulated Terminal Stock Prices ($S_T$)')
axes[0].set_xlabel('$S_T$')
axes[0].set_ylabel('Frequency')
axes[0].legend()

axes[1].hist(payoffs_base, bins=60, color='lightgreen', edgecolor='black', alpha=0.7)
axes[1].set_title('Simulated Call Option Payoffs')
axes[1].set_xlabel('Payoff Value')
axes[1].set_ylabel('Frequency')
axes[1].set_yscale('log')

plt.tight_layout()
plt.show()

print("-" * 40)

sigmas = [0.1, 0.3, 0.5]
print("Repeating simulation for different volatilities:\n")

for sigma in sigmas:
    ST = S0 * np.exp(sigma * Z)
    payoffs = np.maximum(ST - K, 0)
    expected_payoff = np.mean(payoffs)
    print(f"Sigma = {sigma}: Estimated Expected Payoff = {expected_payoff:.4f}")