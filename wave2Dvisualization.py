"""Method contains Solver1D class for simulating PDEs on 1D grids."""
import nengo
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import interp1d
import gc




def feedback_connection(u):
    c=4.2
    return np.array((u[1], -2*c**2/dx**2 * u[0]))


def lateral_connection(u):
    c=4.2
    return np.array((0, c**2/dx**2 * u[0]))

def population_values():  # [grid index, time step]
    return np.swapaxes(grid_values, 0, 1)

def _feedback_update(u):
    """Update equation for the feedback connection."""
    return u + dt*feedback_connection(u)

def _lateral_update(u):
    """Update equation for the lateral connection"""
    return dt*lateral_connection(u)


"""Run simulation using Nengo framework.

Args:
    dt (float): Time step of PDE.
    t_steps (int): Number of time steps in simulation.
    x_steps (int): Number of spatial steps in simulation.
    boundaries (float): Boundary conditions.
    neurons (int): Number of neurons used per population.
    radius (float): The radius for training the neuron populations.
"""


# Nengo simulation
t_steps = 100  # Number of time steps
x_steps = 20  # Number of x steps
neurons = 500
radius = 14

# Grid properties
K = 4.2
x_len = 80  # mm
dx = x_len/x_steps
dt = dx**2/(2*K**2)  # dt chosen for stability

def boundaries(t):
    return[10 * np.exp(-(t / dt - 10) ** 2 / 20),
           10 * np.exp(-(t / dt - 10) ** 2 / 20) * -2 * (t / dt - 10) / 20 / dt]

for _ in range(5000):

    x_steps += 5
    print(f"Neurons: {neurons} - Grid Size: {x_steps}x{x_steps}")
    model = nengo.Network()
    with model:
        states = []
        state_probes = []
        for i in range(x_steps):
            states.append([nengo.Ensemble(neurons, 2, radius) for _ in range(x_steps)])
            state_probes.append([nengo.Probe(state, synapse=dt) for state in states[i]])

        for i in range(x_steps):
            for j in range(x_steps):
                if i == 0:
                    nengo.Connection(nengo.Node(boundaries),
                                     states[0][j], dt, _lateral_update)
                else:
                    nengo.Connection(states[i][j], states[i-1][j], dt, _lateral_update)
                if i == x_steps-1:
                    nengo.Connection(nengo.Node(boundaries),
                                     states[-1][j], dt, _lateral_update)
                else:
                    nengo.Connection(states[i][j], states[i+1][j], dt, _lateral_update)

                if j == 0:
                    nengo.Connection(nengo.Node(boundaries),
                                     states[i][0], dt, _lateral_update)
                else:
                    nengo.Connection(states[i][j], states[i][j-1], dt, _lateral_update)
                if j == x_steps-1:
                    nengo.Connection(nengo.Node(boundaries),
                                     states[i][-1], dt, _lateral_update)
                else:
                    nengo.Connection(states[i][j], states[i][j+1], dt, _lateral_update)

                nengo.Connection(states[i][j], states[i][j], dt, _feedback_update)

    with nengo.Simulator(model, dt=0.001) as sim:
        sim.run(t_steps * dt)

        # Step 1: Extract data into a 3D array (time, x, y)
    grid_values = np.zeros((t_steps, x_steps, x_steps))  # Array to hold grid states over time

    for i in range(x_steps):
        for j in range(x_steps):
            probe_data = sim.data[state_probes[i][j]].flatten()

            # Verify lengths of time and probe data
            probe_time = np.linspace(0, t_steps * dt, len(probe_data))
            target_time = np.linspace(0, t_steps * dt, t_steps)

            if len(probe_time) != len(probe_data):
                print(
                    f"Warning: Length mismatch at ({i}, {j}). Probe time: {len(probe_time)}, Probe data: {len(probe_data)}")

            # Interpolate data if the lengths match
            if len(probe_time) == len(probe_data):
                interp_func = interp1d(probe_time, probe_data, kind="linear", fill_value="extrapolate")
                grid_values[:, i, j] = interp_func(target_time)
            else:
                # Handle the case where lengths don't match (you may want to handle this differently)
                print(f"Skipping interpolation for ({i}, {j}) due to length mismatch.")

    # Step 2: Set up the plot
    # Step 2: Set up the plot with fixed color scaling
    fig, ax = plt.subplots()

    # Set the vmin and vmax based on the range of the first frame
    vmin = np.min(grid_values)
    vmax = np.max(grid_values)

    # Create the heatmap with fixed vmin and vmax
    heatmap = ax.imshow(grid_values[0], cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title("Wave Propagation on 2D Grid")
    plt.colorbar(heatmap)



    # Step 3: Update function for animation
    def update(frame):
        heatmap.set_data(grid_values[frame])
        print(f"Frame {frame:.2f} done.")
        ax.set_title(f"Wave Propagation {x_steps}x{x_steps} - {neurons} Neurons - Time Step {frame}")
        return heatmap,


    # Step 4: Create the animation
    ani = animation.FuncAnimation(fig, update, frames=t_steps, interval=50, blit=True)

    # Step 5: Show the animation

    ani.save(f"wave_propagation_{x_steps}x{x_steps}_{neurons}neurons.mp4", fps=20, extra_args=['-vcodec', 'libx264'])

    # Freigabe von nicht mehr benötigten Variablen
    del model, sim, grid_values, fig, ax, ani

    # Garbage Collector aufrufen
    gc.collect()







