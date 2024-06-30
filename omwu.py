import numpy as np
import matplotlib.pyplot as plt

#def omwu_update(weights, previous_payoffs, current_payoffs, epsilon):
#    optimistic_weights = weights * np.exp(epsilon * previous_payoffs)
#    optimistic_weights /= np.sum(optimistic_weights)
#    corrected_weights = optimistic_weights * np.exp(epsilon * (current_payoffs - previous_payoffs))
#    corrected_weights /= np.sum(corrected_weights)
#    return corrected_weights


def omwu_update(weights, previous_payoffs, current_payoffs, epsilon):
    combined_payoffs = 1 + epsilon * (current_payoffs + previous_payoffs)
    updated_weights = weights * combined_payoffs
    return updated_weights

def run_omwu_experiment(epsilon_func, num_iterations=2500):
    weights_row = np.array([0.51, 0.49])
    weights_col = np.array([0.49, 0.51])
    
    #payoff_matrix = np.array([[1, -1], [-1, 1]])
    # for coordination game
    payoff_matrix = np.array([[1, 0], [0, 2]])

    previous_payoffs_row = np.zeros(2)
    previous_payoffs_col = np.zeros(2)
    
    history = []
    
    for t in range(1, num_iterations + 1):
        epsilon = epsilon_func(t)        
        p_row = weights_row / np.sum(weights_row)
        p_col = weights_col / np.sum(weights_col)
        
        current_payoffs_row = np.dot(payoff_matrix, p_col)
        #current_payoffs_col = -np.dot(p_row, payoff_matrix)
        # for coordination game
        current_payoffs_col = np.dot(p_row, payoff_matrix)

        weights_row = omwu_update(weights_row, previous_payoffs_row, current_payoffs_row, epsilon)
        weights_col = omwu_update(weights_col, previous_payoffs_col, current_payoffs_col, epsilon)
        
        previous_payoffs_row = current_payoffs_row
        previous_payoffs_col = current_payoffs_col
        
        history.append((p_row[0], p_col[0]))
    
    return np.array(history)

def plot_spiral(history, epsilon_name, subplot):
    plt.subplot(subplot)
    plt.scatter(history[:, 0], history[:, 1], s=3, alpha=1)
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
    history = run_omwu_experiment(epsilon_func, num_iterations=5000)
    plot_spiral(history, epsilon_name, 141 + i)

plt.tight_layout()
plt.show()