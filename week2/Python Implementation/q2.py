import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def integrand(x):
    return np.exp(-x**2)

true_value, _ = quad(integrand, 0, 1)
print(f"True numerical value: {true_value:.6f}\n")

N_values = [10**2, 10**3, 10**4, 10**5]
estimates = []
errors = []

np.random.seed(42)
for N in N_values:
    U = np.random.uniform(0, 1, N)
    estimate = np.mean(np.exp(-U**2))
    estimates.append(estimate)

    error = np.abs(estimate - true_value)
    errors.append(error)
    
    print(f"N = {N:<7}: Estimate = {estimate:.6f}, Absolute Error = {error:.6e}")

plt.figure(figsize=(8, 5))
plt.plot(N_values, estimates, marker='o', linestyle='-', color='blue', label='Monte Carlo Estimate')
plt.axhline(y=true_value, color='red', linestyle='--', label='True Value')
plt.xscale('log')
plt.xlabel('Number of Samples (N)')
plt.ylabel('Estimate of Integral')
plt.title('Monte Carlo Estimate vs. Sample Size N')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()