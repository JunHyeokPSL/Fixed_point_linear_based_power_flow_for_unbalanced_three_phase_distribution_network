import numpy as np
import matplotlib.pyplot as plt
import os 

# Define the network parameters
y_ll = np.array([[7 - 12j, -1 + 2j, -1 + 2j],
                 [-1 + 2j, 7 - 12j, -1 + 2j],
                 [-1 + 2j, -1 + 2j, 7 - 12j]])  # YLL admittance matrix
v_0 = np.array([1, np.exp(-1j * 2 * np.pi / 3), np.exp(1j * 2 * np.pi / 3)])  # Slack bus voltages
s_y = np.array([1.5 + 0.9j, 1.5 + 0.9j, 1.5 + 0.9j])  # Power injections
s_y0 = np.array([0 + 0j, 0+ 0j, 0+ 0j])

# Initial guess for bus voltages (set to slack bus voltage initially)
v = np.copy(v_0)

# Iterative fixed-point method parameters
tolerance = 1e-6
max_iterations = 100

# Store voltage history for visualization
voltage_history = [v]
rho_history = []

# Fixed-point iteration
for iteration in range(max_iterations):
    # Compute the current injection
    i = np.dot(y_ll, v)  # Ohm's law: I = Y * V

    # Update the voltage using the fixed-point formula
    v_new = v_0 + np.linalg.inv(y_ll) @ (np.conj(s_y) / np.conj(v))

    # Store the new voltage for visualization
    voltage_history.append(v_new)

    # Compute the radius rho (distance from the initial condition)
    rho = np.linalg.norm(v_new - v_0, np.inf)
    rho_history.append(rho)

    # Check for convergence
    if np.linalg.norm(v_new - v, np.inf) < tolerance:
        print(f"Converged after {iteration + 1} iterations.")
        break

    v = v_new
else:
    print("Did not converge within the maximum number of iterations.")

# Display the final voltages
print("Final bus voltages:")
print(v)

# Calculate condition (12) related parameters
w = -np.linalg.inv(y_ll) @ y_ll @ v_0
alpha = np.min(np.abs(v_0 / w))  # Alpha from condition (12)
h_matrix = np.array([[1, -1, 0], [0, 1, -1], [-1, 0, 1]])
l_matrix = np.abs(h_matrix)
beta = np.min(np.abs(h_matrix @ v_0) / (l_matrix @ np.abs(w)))


def gamma(v):
    return min(alpha, beta)  # Using the minimum of alpha and beta as per condition (12)

def xi(s):
    # Define W, L, and H matrices
    w_matrix = np.eye(len(s))  # W matrix as identity for simplicity
    h_matrix = np.array([[1, -1, 0], [0, 1, -1], [-1, 0, 1]])  # Example H matrix
    l_matrix = np.abs(h_matrix)  # L matrix as the absolute value of h_matrix
    h_t_matrix = h_matrix.T

    # Calculate xi_Y(s)
    xi_y_s = np.linalg.norm(np.dot(np.dot(np.dot(np.linalg.inv(w_matrix), np.linalg.inv(y_ll)), np.linalg.inv(w_matrix)), s), ord=np.inf)

    # Calculate total xi(s)
    return xi_y_s

gamma_value = gamma(v_0)
xi_s0 = xi(s_y0)
xi_s = xi(s_y)

rho_star = 0.5 * ((gamma_value ** 2 - xi_s0) / gamma_value)  # (14a)

xi_s_diff = xi(s_y - s_y0)
rho_dagger = rho_star - np.sqrt(rho_star**2 - xi_s_diff)  # (14b)


# Calculate parameters for (14a) and (14b)
# (14a) relates to voltage feasibility bounds
v_feasibility_min = 0.95
v_feasibility_max = 1.05

# (14b) relates to system stability bounds - assumed as example values
stability_radius = 0.8

# Plot voltage space and the condition (12) boundary
voltage_history = np.array(voltage_history)
plt.figure(figsize=(10, 6))

# Plot voltage space trajectory
phase = 0
plt.plot(np.real(voltage_history[:, phase]), np.imag(voltage_history[:, phase]), marker='o', label=f'Phase {phase + 1}')

# Plot condition (12) boundary
condition_12_radius = min(alpha, beta)
condition_12_circle = plt.Circle((np.real(v_0[0]), np.imag(v_0[0])), condition_12_radius, color='b', fill=False, linestyle='-', label='Condition (12) Boundary')
plt.gca().add_artist(condition_12_circle)


# Plot feasibility bounds
feasibility_circle_min = plt.Circle((0, 0), v_feasibility_min, color='g', fill=False, linestyle='-.', label='Feasibility Bound (Min)')
feasibility_circle_max = plt.Circle((0, 0), v_feasibility_max, color='g', fill=False, linestyle='-.', label='Feasibility Bound (Max)')
plt.gca().add_artist(feasibility_circle_min)
plt.gca().add_artist(feasibility_circle_max)

# Plot stability bound (centered at 1.0 p.u., per (14a))
feasibility_circle_14a = plt.Circle((1.0, 0), rho_star, color='c', fill=False, linestyle='-.', label='Feasibility Bound (14a)')
plt.gca().add_artist(feasibility_circle_14a)

# Plot stability bound (14b)
stability_circle_14b = plt.Circle((1.0, 0), rho_dagger, color='m', fill=False, linestyle=':', label='Stability Bound (14b)')
plt.gca().add_artist(stability_circle_14b)


plt.xlabel('Real Part of Voltage')
plt.ylabel('Imaginary Part of Voltage')
plt.title('Voltage Space Trajectory, Condition (12) Boundary, Convergence Domain $D^+$, and Feasibility & Stability Bounds')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.legend()

# Set limits for better visualization
plt.xlim(-2, 2)
plt.ylim(-2, 2)

# Define the save path and save the figure
main_path = os.getcwd()  # Assuming main_path is the current directory
save_path = os.path.join(main_path, "fig", "voltage_space_plot.png")
plt.savefig(save_path)

plt.show()
