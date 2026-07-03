import numpy as np

np.random.seed(42)

def simulate_random_walk(d, num_steps):
    pos = np.zeros(d, dtype=int)
    path = [pos.copy()]
    
    for _ in range(num_steps):
        dim = np.random.randint(0, d)
        step = np.random.choice([-1, 1])
        pos[dim] += step
        path.append(pos.copy())
        
    return np.array(path)

walk_1d = simulate_random_walk(d=1, num_steps=10)
walk_2d = simulate_random_walk(d=2, num_steps=10)
walk_3d = simulate_random_walk(d=3, num_steps=10)

def find_prob(d, num_paths=1000, path_length=1000):
    return_count = 0
    
    for _ in range(num_paths):
        pos = np.zeros(d, dtype=int)
        
        for _ in range(path_length):
            dim = np.random.randint(0, d)
            step = np.random.choice([-1, 1])
            pos[dim] += step
            
            if not np.any(pos):
                return_count += 1
                break
                
    return return_count / num_paths


for d in [1,2,3]:
    proportion = find_prob(d, num_paths=1000, path_length=1000)
    print(f"Dimension {d}: {proportion:.3f}")