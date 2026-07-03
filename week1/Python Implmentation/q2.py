import numpy as np

states=[0, 1, 2, 3]
P=np.array([
    [0.60,0.25,0.05,0.10],
    [0.00,0.50,0.50,0.00],
    [0.00,0.50,0.50,0.00],
    [0.00,0.00,0.00,1.00]
])

def simulate_path(start_state, steps, transition_matrix):
    current_state=start_state
    path=[current_state]
    
    for _ in range(steps):
        next_state = np.random.choice(states, p=transition_matrix[current_state])
        path.append(next_state)
        current_state = next_state
        
    return path

np.random.seed(42)

single_path = simulate_path(start_state=0, steps=100, transition_matrix=P)

print(single_path[:21])
print(f"{single_path[-1]}\n")


n=1000
final_states=[]

for _ in range(n):
    path = simulate_path(start_state=0, steps=100, transition_matrix=P)
    final_states.append(path[-1])

final_states = np.array(final_states)

prop_state_3=np.mean(final_states==3)
prop_class_1_2=np.mean((final_states==1)|(final_states==2))

print("After 1000 simulations:")
print(f"Proportion of paths in state 3 at time 100: {prop_state_3:.4f}")
print(f"Proportion of paths in class {{1, 2}} at time 100: {prop_class_1_2:.4f}")