import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

class NBodySimulation:
    def __init__(self, n, masses, positions, velocities, names):
        self.n = n
        self.G = 6.67430e-11  # gravitational constant
        self.masses = np.array(masses)
        self.positions = np.array(positions)
        self.velocities = np.array(velocities)
        self.names = names

    def gravitational_force(self, pos1, pos2, m1, m2):
        r = pos2 - pos1
        distance = np.linalg.norm(r)
        if distance == 0:
            return np.zeros(3)
        force_magnitude = self.G * m1 * m2 / distance ** 2
        force = force_magnitude * r / distance
        return force

    def total_force(self, i, positions):
        force = np.zeros(3)
        for j in range(self.n):
            if i != j:
                force += self.gravitational_force(positions[i], positions[j], self.masses[i], self.masses[j])
        return force

    def derivatives(self, state, t):
        positions = state[:3*self.n].reshape((self.n, 3))
        velocities = state[3*self.n:].reshape((self.n, 3))
        pos_deriv = velocities.flatten()
        vel_deriv = np.zeros(3*self.n)
        
        for i in range(self.n):
            total_force = self.total_force(i, positions)
            vel_deriv[3*i:3*i+3] = total_force / self.masses[i]
        
        return np.concatenate([pos_deriv, vel_deriv])

    def simulate(self, t_span, dt):
        self.t = np.arange(t_span[0], t_span[1], dt)
        initial_state = np.concatenate([self.positions.flatten(), self.velocities.flatten()])
        
        solution = odeint(self.derivatives, initial_state, self.t)
        
        self.positions_over_time = solution[:, :3*self.n].reshape((len(self.t), self.n, 3))
        self.velocities_over_time = solution[:, 3*self.n:].reshape((len(self.t), self.n, 3))

    def animate_trajectories(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Initialize lines and trails
        lines = [ax.plot([], [], [], 'o-', markersize=5, label=name)[0] for name in self.names]
        trails = [ax.plot([], [], [], '-', lw=1)[0] for _ in self.names]

        ax.set_xlim([-2e11, 2e11])
        ax.set_ylim([-2e11, 2e11])
        ax.set_zlim([-2e11, 2e11])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('N-Body Problem Simulation')
        ax.legend()

        def init():
            for line, trail in zip(lines, trails):
                line.set_data([], [])
                line.set_3d_properties([])
                trail.set_data([], [])
                trail.set_3d_properties([])
            return lines + trails

        def animate(i):
            trail_length = 50  # Number of steps to show in the trail
            for j, (line, trail) in enumerate(zip(lines, trails)):
                # Set the current position
                line.set_data(self.positions_over_time[i, j, 0], self.positions_over_time[i, j, 1])
                line.set_3d_properties(self.positions_over_time[i, j, 2])
                
                # Set the trail
                start = max(0, i - trail_length)
                x = self.positions_over_time[start:i, j, 0]
                y = self.positions_over_time[start:i, j, 1]
                z = self.positions_over_time[start:i, j, 2]
                trail.set_data(x, y)
                trail.set_3d_properties(z)
            
            ax.view_init(30, 0.3 * i)  # Rotate view for better visibility
            return lines + trails

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=len(self.t), interval=20, blit=True)

        plt.show()

# Example usage
if __name__ == "__main__":
    n = 3  # number of bodies
    masses = [1.989e30, 5.972e24, 7.34767309e22]  # Sun, Earth, Moon (in kg)
    positions = [
        [0, 0, 0],  # Sun at origin
        [1.496e11, 0, 0],  # Earth 1 AU from Sun
        [1.496e11 + 3.844e8, 0, 0]  # Moon 384,400 km from Earth
    ]
    velocities = [
        [0, 0, 0],  # Sun stationary
        [0, 29.78e3, 0],  # Earth's orbital velocity
        [0, 29.78e3 + 1.022e3, 0]  # Moon's velocity relative to Earth
    ]
    names = ['Sun', 'Earth', 'Moon']

    sim = NBodySimulation(n, masses, positions, velocities, names)
    sim.simulate((0, 365*24*3600), 24*3600)  # Simulate for 1 year with daily steps
    sim.animate_trajectories()