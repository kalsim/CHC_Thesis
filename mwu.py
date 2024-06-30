import numpy as np
import matplotlib.pyplot as plt

def mwu_update(weights, payoffs, epsilon):
    """Update weights using Multiplicative Weights Update algorithm."""
    #return weights * np.exp(epsilon * payoffs)
    return weights * (1 + epsilon * payoffs)

def run_experiment(epsilon_func, num_iterations=2500):
    """Run the MWU algorithm for matching pennies game."""
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
        p_row = weights_row / np.sum(weights_row)
        p_col = weights_col / np.sum(weights_col)
        
        payoffs_row = np.dot(payoff_matrix, p_col)
        #payoffs_col = -np.dot(p_row, payoff_matrix)
        # for coordination game
        payoffs_col = np.dot(p_row, payoff_matrix)

        epsilon = epsilon_func(t)
        weights_row = mwu_update(weights_row, payoffs_row, epsilon)
        weights_col = mwu_update(weights_col, payoffs_col, epsilon)
        
        history.append((p_row[0], p_col[0]))
    
    return np.array(history)

def plot_spiral(history, epsilon_name, subplot):
    plt.subplot(subplot)
    plt.scatter(history[:, 0], history[:, 1], s=3, alpha=1)
    #plt.plot(history[:, 0], history[:, 1], linewidth=0.5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f"ε = {epsilon_name}")
    #plt.xlabel("P(Heads) for Row")
    #plt.ylabel("P(Heads) for Column")
    plt.axhline(y=0.5, color='r', linestyle='--', linewidth=0.5)
    plt.axvline(x=0.5, color='r', linestyle='--', linewidth=0.5)

epsilon_funcs = [
    lambda t: 0.5,
    lambda t: t**(-1/3),
    lambda t: t**(-1/2),
    lambda t: t**(-2/3)
]

epsilon_names = [
    "0.5",
    "t^(-1/3)",
    "t^(-1/2)",
    "t^(-2/3)"
]

fig = plt.figure(figsize=(16, 4))

for i, (epsilon_func, epsilon_name) in enumerate(zip(epsilon_funcs, epsilon_names)):
    history = run_experiment(epsilon_func)
    plot_spiral(history, epsilon_name, 141 + i)

plt.tight_layout()
plt.show()