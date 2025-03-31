import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
TIME_STEP = 1e6  # Time step for simulation (seconds)
CENTRAL_MASS = 5e27  # Mass of the central body (kg)

# Initialize particles
def initialize_particles(num_particles, mass_range, position_range, velocity_range):
    masses = np.random.uniform(*mass_range, num_particles)
    masses = np.insert(masses, 0, CENTRAL_MASS)  # Add central mass
    positions = np.random.uniform(*position_range, (num_particles, 2))
    positions = np.insert(positions, 0, [0, 0], axis=0)  # Set central mass at origin
    velocities = np.random.uniform(*velocity_range, (num_particles, 2))
    velocities = np.insert(velocities, 0, [0, 0], axis=0)  # Set central mass
    return masses, positions, velocities

# Compute gravitational forces
def compute_forces(masses, positions):
    num_particles = len(masses)
    forces = np.zeros_like(positions)
    for i in range(num_particles):
        for j in range(num_particles):
            if i != j:
                r = positions[j] - positions[i]
                distance = np.linalg.norm(r)
                if distance > 1e-2:  # Avoid division by zero
                    force_magnitude = G * masses[i] * masses[j] / distance**2
                    forces[i] += force_magnitude * r / distance
    return forces

# RK4 step function
def rk4_step(masses, positions, velocities, time_step):
    def acceleration(positions):
        forces = compute_forces(masses, positions)
        return forces / masses[:, np.newaxis]

    # RK4 coefficients
    k1_v = acceleration(positions)
    k1_x = velocities

    k2_v = acceleration(positions + 0.5 * time_step * k1_x)
    k2_x = velocities + 0.5 * time_step * k1_v

    k3_v = acceleration(positions + 0.5 * time_step * k2_x)
    k3_x = velocities + 0.5 * time_step * k2_v

    k4_v = acceleration(positions + time_step * k3_x)
    k4_x = velocities + time_step * k3_v

    # Update positions and velocities
    new_positions = positions + (time_step / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
    new_velocities = velocities + (time_step / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    return new_positions, new_velocities

# Calculate total kinetic energy
def calculate_kinetic_energy(masses, velocities):
    # Kinetic energy: KE = 0.5 * m * v^2
    kinetic_energy = 0.5 * masses * np.sum(velocities**2, axis=1)
    return np.sum(kinetic_energy)

# Update function using RK4
def update(masses, positions, velocities):
    return rk4_step(masses, positions, velocities, TIME_STEP)

# Simulation parameters
NUM_PARTICLES = 15
MASS_RANGE = (1e20, 1e22)  # Mass range in kg
POSITION_RANGE = (-1e11, 1e11)  # Position range in meters
VELOCITY_RANGE = (-2e3, 2e3)  # Velocity range in meters/second

# Initialize particles
masses, positions, velocities = initialize_particles(NUM_PARTICLES, MASS_RANGE, POSITION_RANGE, VELOCITY_RANGE)

# Set up the plot
fig, ax = plt.subplots()
scat = ax.scatter(positions[1:, 0], positions[1:, 1], s=np.log10(masses[1:]), c='blue')
central_mass = ax.scatter(positions[0, 0], positions[0, 1], s=100, c='red', marker='*')
ax.set_xlim((3*value for value in POSITION_RANGE))
ax.set_ylim((3*value for value in POSITION_RANGE))
ax.set_aspect('equal')
ax.set_title("Gravity Simulator")
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")

# Add a text box to display kinetic energy
kinetic_energy_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, color='red')

# Animation update function
def animate(frame):
    global positions, velocities
    positions, velocities = update(masses, positions, velocities)
    scat.set_offsets(positions[1:])
    central_mass.set_offsets(positions[0])
     # Update kinetic energy
    total_kinetic_energy = calculate_kinetic_energy(masses, velocities)
    kinetic_energy_text.set_text(f"Kinetic Energy: {total_kinetic_energy:.2e} J")
    return scat, kinetic_energy_text

# Create animation
ani = FuncAnimation(fig, animate, frames=200, interval=10, blit=True)

if __name__ == "__main__":
    plt.show()