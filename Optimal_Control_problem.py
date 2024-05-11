import numpy as np
import matplotlib.pyplot as plt

## Function to calculate the derivatives of the state variables
def dynamics(state, control_input):
    return state + control_input

# Function to calculate transition costs
def transition_cost(x, u):
    return 0.5 * (x**2 + u**2)

# Function to find the nearest valid state
def nearest_state(x, states):
    return min(states, key=lambda state: abs(state - x))

# Initial conditions and discrete k horizon
Khorizon = 4
x_0 = 2
states = [0, 1, 2, 0.2, 0.4, 1.2]

# Control inputs for each x
u_steps = {
    2: [-1.6, -0.8, 0],
    1: [-0.8, 0],
    0: [0],
    1.2: [0],
    0.4: [0],
    0.2: [0],
}

# Plot setup
fig, ax = plt.subplots(figsize=(10, 6))
plt.xlabel('k')
plt.ylabel('x')
plt.grid(True)
plt.text(0,x_0 + 0.03,"x[0]",color='m')
plt.xticks(range(Khorizon + 1))
plt.yticks(range(int(x_0) + 1))

# Cost to go nested dict
V = {k: {x: np.inf for x in states} for k in range(1, Khorizon)}
V[0] = {x_0: np.inf}                                                    # V[0] only contains the initial state x_0
V[1] = {x: np.inf for x in states if x != 0.2}                          # V[1] does not contain the state 0.2
V[Khorizon] = {0: 0}                                                    # Terminal state

# Main loop through k values
for k in range(Khorizon - 1, -1, -1):
    if k == 0:
        x_values = [x_0]                                                # Only start from x_0 at k=0
    elif k == 1:
        x_values = [0, 1, 2, 0.4, 1.2]
    else:
        x_values = states

    for x in x_values:
        plt.plot(k, x, 'bo')

    for x in x_values:
        if k == Khorizon - 1:
            x_next = 0
            cost = transition_cost(x, -x)
            if x in [0, 1, 2]:
                V[k][x] = cost
            elif x in [0.2, 0.4]:
                V[k][x] = V[k][0] + (V[k][1]-V[k][0]) * x
            elif x in [1.2]:
                V[k][x] = V[k][1] + (V[k][2] - V[k][1]) * (x -1)
            plt.plot([k, k + 1], [x, x_next], 'b--')
            plt.plot(k + 1, x_next, 'bo')
            plt.text(k + 0.5, x / 2, f"{cost:.2f}", color='red')
        else:
            u_values = u_steps.get(x, [0])
            for u in u_values:
                x_next = dynamics(x, u)
                x_next = nearest_state(x_next, states)         # Ensure the next state is valid
                cost = transition_cost(x, u)
                if V[k+1][x_next] + cost < V[k][x]:
                  if x in [0, 1, 2]:
                    V[k][x] = V[k+1][x_next] + cost
                  elif x in [0.2, 0.4]:
                    V[k][x] = V[k][0] + (V[k][1]-V[k][0]) * x
                  elif x in [1.2]:
                    V[k][x] = V[k][1] + (V[k][2] - V[k][1]) * (x -1)
                color = 'bo' if x_next in [0, 1, 2] else 'ro'
                plt.plot([k, k + 1], [x, x_next], 'b--')
                plt.plot(k + 1, x_next, color)
                if x in [0.4, 1.2]:                                                              #overlaping text fix
                    plt.text(k + 0.2, (x + x_next) / 1.98, f"{cost:.2f}", color='red')
                else:
                    plt.text(k + 0.5, (x + x_next) / 1.98, f"{cost:.2f}", color='red')

# Initialize the path with the starting state
optimal_path = [x_0]
current_state = x_0

# Plotting the optimal path with feasible transitions
for k in range(Khorizon):
    if current_state not in u_steps:                         # If no more controls are available, break out of the loop
        break

    feasible_transitions = {}
    for u in u_steps[current_state]:
        if k == 3:
            next_state = dynamics(current_state, - current_state)
        else:
            next_state = dynamics(current_state, u)
        next_state = nearest_state(next_state, states)
        if next_state in V[k+1]:                                                   # Check if the next state is part of the valid future states
            feasible_transitions[next_state] = V[k+1][next_state]

    if not feasible_transitions:                         # If no feasible transitions, break out of the loop
        break

    # Select the state with the minimum cost-to-go from the feasible transitions
    next_state = min(feasible_transitions, key=feasible_transitions.get)
    optimal_path.append(next_state)
    current_state = next_state

    # Plotting the transition
    if k == 3:
        plt.plot([k, k+1], [optimal_path[-2], 0], 'r->', linewidth=2, label='Optimal Path' if k == 0 else "")
    else:
        plt.plot([k, k+1], [optimal_path[-2], optimal_path[-1]], 'r->', linewidth=2, label='Optimal Path' if k == 0 else "")



plt.legend()
plt.show()


print("\nOptimal path:", ' -> '.join(map(str, optimal_path)))
print(f"Minimum cost-to-go from initial state x_0 = {x_0}: {V[0][x_0]:.2f}")
print("\n----------------------------------------------------------")

# Print the entire dictionary V
for k in sorted(V.keys()):
    print(f"Time step {k}:")
    for x in sorted(V[k].keys()):
        print(f"  State {x}: Cost-to-go = {V[k][x]:.2f}")
    print()


