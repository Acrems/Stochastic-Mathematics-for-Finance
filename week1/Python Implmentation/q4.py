import numpy as np
import matplotlib.pyplot as plt

states=[0,1,2,3]

P=np.array([
    [0.60,0.30,0.10,0.00],
    [0.20,0.50,0.20,0.10],
    [0.10,0.30,0.40,0.20],
    [0.05,0.15,0.30,0.50]
])

returns=np.array([0.012, 0.004, -0.006, -0.025])

np.random.seed(42)
n_weeks=10000
path=np.zeros(n_weeks, dtype=int)

current_state = 0 

for t in range(n_weeks):
    current_state = np.random.choice(states, p=P[current_state])
    path[t] = current_state

print(f"First 10 weeks: {path[:10]}\n")

A = P.T - np.eye(4)
A[-1] = np.ones(4)
b = np.array([0, 0, 0, 1])
theoretical_pi=np.linalg.solve(A, b)

empirical_pi = np.array([np.mean(path == s) for s in states])

print("Regime   | Empirical Fraction | Theoretical Stationary")
print("------------------------------------------------------")
regime_names = ["Bull", "Neutral", "Bear", "Crisis"]
for i in range(4):
    print(f"{regime_names[i]:<8} | {empirical_pi[i]:.4f} | {theoretical_pi[i]:.4f}")
print()

empirical_avg_return = np.mean(returns[path])

theoretical_avg_return = np.dot(theoretical_pi, returns)
print(f"Empirical Average Weekly Return   : {empirical_avg_return:.6f}")
print(f"Theoretical Average Weekly Return : {theoretical_avg_return:.6f}\n")

simulated_returns = returns[path]
cumulative_returns = np.cumsum(simulated_returns)

plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns, color='blue', linewidth=1)
plt.title('Cumulative Portfolio Return Over 10,000 Weeks', fontsize=14)
plt.xlabel('Weeks (n)', fontsize=12)
plt.ylabel('Cumulative Return ($C_n$)', fontsize=12)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()