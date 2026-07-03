import numpy as np

np.random.seed(42)

#probability for the states:(0,0),(1,0),(0,1),(1,1)
probabilities = [0.77, 0.03, 0.13, 0.07]
states=[(0,0),(1,0),(0,1),(1,1)]
n=10000

indices=np.random.choice(len(states),size=n,p=probabilities)

I_A=np.array([states[i][0] for i in indices])
I_B=np.array([states[i][1] for i in indices])

#for i in range(10):
#   print(f"({I_A[i]}, {I_B[i]})")

cnt = {(0, 0):0,(1, 0):0,(0, 1): 0,(1, 1):0}

for i in range(n):
    state=(I_A[i],I_B[i])
    cnt[state]+=1

for state,count in cnt.items():
    empirical_prob = count / n
    print(f"State(I_A={state[0]},I_B={state[1]}):")
    print(f"Empirical Prob:{empirical_prob:.4f}\n")