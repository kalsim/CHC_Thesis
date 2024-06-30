import numpy as np
import matplotlib.pyplot as plt

def ogd_update(weights, payoffs, learning_rate):
    return weights + learning_rate * payoffs

def project_simplex(v):
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u > (cssv - 1) / np.arange(1, len(v) + 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)

def run_experiment(learning_rate_func, num_iterations=2500):
    strategies = ['Heads', 'Tails']
    num_strategies = len(strategies)
    
    weights_row = np.array([0.51, 0.49])
    weights_col = np.array([0.49, 0.51])
    
    # Payoff matrix for matching pennies
    #payoff_matrix = np.array([[1, -1], [-1, 1]])
    # for coordination game
    payoff_matrix = np.array([[1, 0], [0, 2]])
    
    history = []
    
    for t in range(1, num_iterations + 1):
        p_row = project_simplex(weights_row)
        p_col = project_simplex(weights_col)
        
        payoffs_row = np.dot(payoff_matrix, p_col)
        #payoffs_col = -np.dot(p_row, payoff_matrix)
        # for coordination game
        payoffs_col = np.dot(p_row, payoff_matrix)
        
        learning_rate = learning_rate_func(t)
        weights_row = ogd_update(weights_row, payoffs_row, learning_rate)
        weights_col = ogd_update(weights_col, payoffs_col, learning_rate)
        
        history.append((p_row[0], p_col[0]))
    
    return np.array(history)

def plot_spiral(history, learning_rate_name, subplot):
    plt.subplot(subplot)
    plt.scatter(history[:, 0], history[:, 1], s=3, alpha=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f"Learning Rate = {learning_rate_name}")
    #plt.xlabel("P(Heads) for Row")
    #plt.ylabel("P(Heads) for Column")
    plt.axhline(y=0.5, color='r', linestyle='--', linewidth=0.5)
    plt.axvline(x=0.5, color='r', linestyle='--', linewidth=0.5)

learning_rate_funcs = [
    lambda t: 0.5,
    lambda t: t**(-1/3),
    lambda t: t**(-1/2),
    lambda t: t**(-2/3)
]

learning_rate_names = [
    "0.5",
    "t^(-1/3)",
    "t^(-1/2)",
    "t^(-2/3)"
]

fig = plt.figure(figsize=(16, 4))

for i, (learning_rate_func, learning_rate_name) in enumerate(zip(learning_rate_funcs, learning_rate_names)):
    history = run_experiment(learning_rate_func)
    plot_spiral(history, learning_rate_name, 141 + i)

plt.tight_layout()
plt.show()