import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

S0 = 100; mu = 0.07; T = 1.0; N = 500; n_paths = 50000
dt = T / N

def simulate_gbm(sigma):
    S = np.full(n_paths, S0, dtype=float)
    for _ in range(N):
        dW = np.random.normal(0, np.sqrt(dt), n_paths)
        S += mu * S * dt + sigma * S * dW # Euler-Maruyama step
    return S
sigma = 0.2
S1 = simulate_gbm(sigma)

emp_mean = np.mean(S1)
emp_std = np.std(S1)
theo_mean = S0 * np.exp(mu * T)
theo_std = S0 * np.exp(mu * T) * np.sqrt(np.exp(sigma**2 * T) - 1)

print(f"(e) Mean: Emp = {emp_mean:.2f}, Theo = {theo_mean:.2f}")
print(f"(e) Std:  Emp = {emp_std:.2f}, Theo = {theo_std:.2f}\n")

# Plot Histogram vs Lognormal PDF
plt.hist(S1, bins=100, density=True, alpha=0.6, label='Empirical (Euler-Maruyama)')
x = np.linspace(min(S1), max(S1), 500)
scale = S0 * np.exp(mu * T)
s = sigma * np.sqrt(T) # shape parameter for scipy lognorm
pdf = lognorm.pdf(x, s, scale=S0 * np.exp((mu - sigma**2/2)*T))
plt.plot(x, pdf, 'r-', lw=2, label='Theoretical Lognormal PDF')
plt.legend()
plt.title('Terminal Stock Price Distribution S_1')
plt.show()

K = 110
print("(f) Volatility Analysis:")
for sig in [0.1, 0.3, 0.5]:
    S_sig = simulate_gbm(sig)
    payoff = np.maximum(S_sig - K, 0)
    print(f"Sigma {sig}: Mean={np.mean(S_sig):.2f}, Std={np.std(S_sig):.2f}, E[Payoff]={np.mean(payoff):.2f}")
    
log_rets = np.log(S1 / S0)
emp_lr_mean = np.mean(log_rets)
emp_lr_var = np.var(log_rets)

theo_lr_mean = (mu - sigma**2 / 2) * T
theo_lr_var = (sigma**2) * T

print(f"\n(g) Log-Returns:")
print(f"Empirical Mean = {emp_lr_mean:.4f}, Theoretical = {theo_lr_mean:.4f}")
print(f"Empirical Var  = {emp_lr_var:.4f}, Theoretical = {theo_lr_var:.4f}")