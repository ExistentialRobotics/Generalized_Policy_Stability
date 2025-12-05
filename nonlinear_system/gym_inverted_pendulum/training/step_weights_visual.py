import torch
import numpy as np
import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from network.PolicyNet import StepNet
from network.lyapunov_net import LyapunovNet


class StepNetVisualizer:
    def __init__(
        self,
        model_path,
        stepnet_path,
        residual_path,
        algo_type="ppo",
        n_steps=20,
        hidden_dim=64,
        n_layers=3,
        results_dir=None,
        beta=0.01,
    ):
        # Load RL model based on algorithm type
        self.algo_type = algo_type.lower()
        if self.algo_type == "ppo":
            from stable_baselines3 import PPO

            self.rl_model = PPO.load(model_path)
        elif self.algo_type == "sac":
            from stable_baselines3 import SAC

            self.rl_model = SAC.load(model_path)
        else:
            raise ValueError(f"Unsupported algorithm type: {algo_type}")

        self.env = gym.make("Pendulum-v1")

        # Initialize networks
        self.stepnet = StepNet(n_input=3, n_hidden=hidden_dim, n_steps=n_steps, n_layers=n_layers)
        self.residual_net = LyapunovNet(n_input=3, n_hidden=hidden_dim, n_layers=n_layers)

        # Load pre-trained weights
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stepnet.load_state_dict(torch.load(stepnet_path, map_location=self.device))
        self.residual_net.load_state_dict(torch.load(residual_path, map_location=self.device))

        self.stepnet.to(self.device)
        self.residual_net.to(self.device)
        self.n_steps = n_steps
        self.results_dir = results_dir if results_dir else os.getcwd()
        self.beta = beta

        # Find practical equilibrium state
        self.equilibrium_state, self.value_at_equilibrium = self.find_practical_equilibrium()
        print("\nPractical equilibrium found:")
        print(
            f"State: theta={np.arctan2(self.equilibrium_state[1], self.equilibrium_state[0]):.6f} rad, "
            f"omega={self.equilibrium_state[2]:.6f} rad/s"
        )
        print(f"Value at equilibrium: {self.value_at_equilibrium:.6f}")

    def get_value(self, obs_tensor):
        """Get value from RL model based on algorithm type"""
        with torch.no_grad():
            if self.algo_type == "ppo":
                value = float(self.rl_model.policy.predict_values(obs_tensor).cpu().numpy().item())
            else:  # SAC
                # Get action from actor for value computation
                action = self.rl_model.actor(obs_tensor)[0]
                action = action.unsqueeze(0)
                # Get minimum Q-value
                q1, q2 = self.rl_model.critic(obs_tensor, action)
                value = float(torch.min(q1, q2).cpu().numpy().item())
        return value

    def find_practical_equilibrium(self, simulation_steps=200):
        """Simulate system from upright position to find practical equilibrium"""
        self.env.reset()
        self.env.unwrapped.state = np.array([0.0, 0.0])
        obs = np.array([1.0, 0.0, 0.0])

        states = []
        values = []

        for _ in range(simulation_steps):
            states.append(obs)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            value = self.get_value(obs_tensor)
            action, _ = self.rl_model.predict(obs, deterministic=True)
            values.append(value)
            obs, _, _, _, _ = self.env.step(action)

        last_n = 10
        equilibrium_state = np.mean(states[-last_n:], axis=0)
        equilibrium_value = np.mean(values[-last_n:])

        return equilibrium_state, equilibrium_value

    def get_residual_value(self, state):
        """
        Compute the Lyapunov function value with the corrected construction:
        V(x) = |V_RL(x) - V_RL(x*)| + ‖φ(x) - φ(x*)‖² + α‖x - x*‖²
        """
        # Get RL value for current state
        with torch.no_grad():
            if self.algo_type == "ppo":
                v_rl = self.rl_model.policy.predict_values(state)
            else:  # SAC
                action = self.rl_model.actor(state)[0]
                q1, q2 = self.rl_model.critic(state, action.unsqueeze(0))
                v_rl = torch.min(q1, q2)

        # Compute φ(x) for current state
        phi_x = self.residual_net(state)

        # Get equilibrium values
        equilibrium_tensor = torch.FloatTensor(self.equilibrium_state).to(self.device)
        phi_equilibrium = self.residual_net(equilibrium_tensor)

        # Compute the three terms
        # Term 1: |V_RL(x) - V_RL(x*)|
        term1 = torch.abs(v_rl - self.value_at_equilibrium)

        # Term 2: ‖φ(x) - φ(x*)‖²
        term2 = torch.pow(torch.norm(phi_x - phi_equilibrium), 2)

        # Term 3: β‖x - x*‖²
        state_diff = state - equilibrium_tensor
        term3 = self.beta * torch.pow(torch.norm(state_diff), 2)

        return term1 + term2 + term3

    def collect_trajectory(self, initial_state, max_steps=200):
        """Collect trajectory using RL policy"""
        theta, omega = initial_state
        self.env.reset()
        self.env.unwrapped.state = np.array([theta, omega])
        obs = np.array([np.cos(theta), np.sin(theta), omega])

        states = []
        values = []

        for _ in range(max_steps):
            states.append(obs)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            value = self.get_value(obs_tensor)
            action, _ = self.rl_model.predict(obs, deterministic=True)

            values.append(-value)
            next_obs, _, done, _, _ = self.env.step(action)
            obs = next_obs

            if done:
                break

        return np.array(states), np.array(values)

    def is_near_equilibrium(self, state, threshold=0.02):
        """Check if a state is close to the equilibrium"""
        theta = np.arctan2(state[1], state[0])
        omega = state[2]
        eq_theta = np.arctan2(self.equilibrium_state[1], self.equilibrium_state[0])
        eq_omega = self.equilibrium_state[2]

        # Check if both theta and omega are within threshold
        return abs(theta - eq_theta) < threshold and abs(omega - eq_omega) < threshold

    def plot_trajectories_and_values(self, num_trajectories=5, num_steps=200, seed=42):
        """Plot trajectories and their corresponding value functions (both original and new)"""
        np.random.seed(seed)
        torch.manual_seed(seed)

        all_trajectories = []
        all_original_values = []
        all_new_values = []
        all_value_diffs = []  # New list to store value differences

        for i in range(num_trajectories):
            # Random initial state
            theta = np.random.uniform(-np.pi, np.pi)
            omega = np.random.uniform(-8, 8)

            # Set initial state
            obs = np.array([np.cos(theta), np.sin(theta), omega])
            self.env.reset()
            self.env.unwrapped.state = np.array([theta, omega])

            trajectory = [[theta, omega]]
            original_values = []
            new_values = []
            value_diffs = []  # Store value differences for this trajectory

            # Get initial values and weights
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                sigma = self.stepnet(obs_tensor)  # Get weights for future values
                original_value = float(self.get_value(obs_tensor))
                residual = self.get_residual_value(obs_tensor)
                initial_total = float(residual.cpu().numpy().item())

            original_values.append(-original_value)
            new_values.append(initial_total)

            # Run trajectory
            current_theta = theta
            states = [obs]  # Store states for computing future values

            for step in range(num_steps):
                action, _ = self.rl_model.predict(obs, deterministic=True)
                obs, _, done, _, _ = self.env.step(action)
                states.append(obs)

                new_theta = np.arctan2(obs[1], obs[0])
                # Adjust theta for continuity
                if new_theta - current_theta > np.pi:
                    new_theta -= 2 * np.pi
                elif new_theta - current_theta < -np.pi:
                    new_theta += 2 * np.pi
                current_theta = new_theta

                omega = obs[2]
                trajectory.append([current_theta, omega])

                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    original_value = float(self.get_value(obs_tensor))
                    residual = self.get_residual_value(obs_tensor)
                    total_value = float(residual.cpu().numpy().item())

                original_values.append(-original_value)
                new_values.append(total_value)

                # Compute value difference at each step
                future_values = []
                for future_state in states[step + 1 : step + 1 + self.n_steps]:
                    state_tensor = torch.FloatTensor(future_state).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        residual = self.get_residual_value(state_tensor)
                        future_total = float(residual.cpu().numpy().item())
                        future_values.append(future_total)

                # Compute weighted average using StepNet weights
                if len(future_values) > 0:
                    future_values = torch.FloatTensor(future_values).to(self.device)
                    weighted_sum = (
                        torch.sum(sigma[0, : len(future_values)] * future_values) / self.n_steps
                    )
                    avg_future_value = float(weighted_sum.cpu().numpy())
                    value_diff = avg_future_value - total_value
                else:
                    value_diff = 0.0

                value_diffs.append(value_diff)

                if done:
                    break

            all_trajectories.append(np.array(trajectory))
            all_original_values.append(np.array(original_values))
            all_new_values.append(np.array(new_values))
            all_value_diffs.append(np.array(value_diffs))

        # Create plots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

        # Plot trajectories
        ax1.set_title("State Space Trajectories", fontsize=14)
        ax1.set_xlabel("Theta (radians)", fontsize=12)
        ax1.set_ylabel("Omega (rad/s)", fontsize=12)

        colors = plt.cm.rainbow(np.linspace(0, 1, num_trajectories))
        for traj, color in zip(all_trajectories, colors):
            ax1.plot(traj[:, 0], traj[:, 1], c=color, alpha=0.7)
            ax1.plot(traj[0, 0], traj[0, 1], "go", markersize=8)  # Start point
            ax1.plot(traj[-1, 0], traj[-1, 1], "ro", markersize=8)  # End point

        # Plot value functions
        ax2.set_title("Value Function Evolution", fontsize=14)
        ax2.set_xlabel("Time Step", fontsize=12)
        ax2.set_ylabel("Value", fontsize=12)

        for orig_values, new_values, color in zip(all_original_values, all_new_values, colors):
            ax2.plot(orig_values, c=color, linestyle="--", alpha=0.7)
            ax2.plot(new_values, c=color, linestyle="-", alpha=0.7)

        # Add legend for the first trajectory only
        lines = ax2.get_lines()[:2]
        labels = ["Original Value", "New Value"]
        ax2.legend(lines, labels)

        # Plot value differences
        ax3.set_title("Value Differences Evolution", fontsize=14)
        ax3.set_xlabel("Time Step", fontsize=12)
        ax3.set_ylabel("Value Difference", fontsize=12)

        for diffs, color in zip(all_value_diffs, colors):
            ax3.plot(diffs, c=color, alpha=0.7)
            # Add horizontal line at y=0 to show the boundary between increase and decrease
            ax3.axhline(y=0, color="k", linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "value_comparison.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def plot_value_differences(
        self, theta_bins=50, omega_bins=50, num_trajectories=10, max_steps=200
    ):
        """Plot the difference between future and current values across state space with trajectories"""
        theta = np.linspace(-np.pi * 2, np.pi * 2, theta_bins)
        omega = np.linspace(-8, 8, omega_bins)
        Theta, Omega = np.meshgrid(theta, omega)

        V_diff_values = np.zeros_like(Theta)
        V_total_values = np.zeros_like(Theta)  # For storing total values

        # First compute the Lyapunov function values
        for i in range(theta_bins):
            for j in range(omega_bins):
                theta_val = theta[i]
                omega_val = omega[j]

                # Get initial state and value
                obs = np.array([np.cos(theta_val), np.sin(theta_val), omega_val])
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

                # Get weights from StepNet for this state
                with torch.no_grad():
                    sigma = self.stepnet(obs_tensor)
                    V_initial = self.get_residual_value(obs_tensor)

                # Store the total value
                V_total_values[j, i] = V_initial.item()

                # Collect trajectory
                states, _ = self.collect_trajectory([theta_val, omega_val], max_steps=self.n_steps)

                # Compute weighted average of future values
                future_values = []
                for state in states[1 : self.n_steps + 1]:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        V = self.get_residual_value(state_tensor)
                        future_values.append(V.item())

                # Compute weighted average using StepNet weights
                if len(future_values) > 0:
                    future_values = torch.FloatTensor(future_values).to(self.device)
                    weighted_sum = (
                        torch.sum(sigma[0, : len(future_values)] * future_values) / self.n_steps
                    )
                    avg_future_value = float(weighted_sum.cpu().numpy())
                else:
                    avg_future_value = V_initial.item()

                V_diff_values[j, i] = avg_future_value - V_initial.item()

        # Create figure for total Lyapunov function with trajectories
        plt.figure(figsize=(10, 8))

        # Plot the contour of total values
        contour1 = plt.contourf(Theta, Omega, V_total_values, levels=50, cmap="viridis", alpha=0.7)
        cbar1 = plt.colorbar(contour1)
        cbar1.ax.tick_params(labelsize=24)

        # Generate and plot trajectories
        colors = plt.cm.rainbow(np.linspace(0, 1, num_trajectories))
        for i in range(num_trajectories):
            # Random initial state
            theta = np.random.uniform(-np.pi, np.pi)
            omega = np.random.uniform(-8, 8)

            # Set initial state
            obs = np.array([np.cos(theta), np.sin(theta), omega])
            self.env.reset()
            self.env.unwrapped.state = np.array([theta, omega])

            # Store trajectory
            trajectory = [[theta, omega]]

            # Run trajectory
            for _ in range(max_steps):
                action, _ = self.rl_model.predict(obs, deterministic=True)
                obs, _, done, _, _ = self.env.step(action)

                new_theta = np.arctan2(obs[1], obs[0])
                # Adjust theta for continuity
                if new_theta - trajectory[-1][0] > np.pi:
                    new_theta -= 2 * np.pi
                elif new_theta - trajectory[-1][0] < -np.pi:
                    new_theta += 2 * np.pi

                trajectory.append([new_theta, obs[2]])

                if done:
                    break

            trajectory = np.array(trajectory)

            # Clip theta values to be within [-2π, 2π]
            trajectory[:, 0] = np.clip(trajectory[:, 0], -2 * np.pi, 2 * np.pi)

            # Plot trajectory
            plt.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                "-",
                color=colors[i],
                linewidth=2,
                alpha=0.8,
                label=f"Trajectory {i + 1}",
            )
            # Plot start point
            plt.plot(
                trajectory[0, 0], trajectory[0, 1], "o", color=colors[i], markersize=8, alpha=0.8
            )

        # Plot equilibrium point
        eq_theta = np.arctan2(self.equilibrium_state[1], self.equilibrium_state[0])
        eq_omega = self.equilibrium_state[2]
        plt.plot(eq_theta, eq_omega, "k*", markersize=15, label="Equilibrium")

        plt.xlabel("Theta (radians)", fontsize=24)
        plt.ylabel("Theta_dot (rad/s)", fontsize=24)
        plt.tick_params(axis="both", which="major", labelsize=22)

        # Set strict x-axis limits
        plt.xlim(-2 * np.pi, 2 * np.pi)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "total_lyapunov_with_trajectories.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # Plot value differences (original plot)
        plt.figure(figsize=(10, 8))
        contour2 = plt.contourf(Theta, Omega, V_diff_values, levels=50, cmap="viridis")
        cbar2 = plt.colorbar(contour2)
        cbar2.ax.tick_params(labelsize=24)

        # Find violation points and plot them as red dots only if there are violations
        violation_mask = V_diff_values > 0
        if np.any(violation_mask):
            violation_thetas = Theta[violation_mask]
            violation_omegas = Omega[violation_mask]
            plt.plot(
                violation_thetas,
                violation_omegas,
                "r.",
                markersize=3,
                alpha=0.5,
                label="Violations",
            )
            plt.legend()

        plt.xlabel("Theta (radians)", fontsize=24)
        plt.ylabel("Theta_dot (rad/s)", fontsize=24)
        plt.tick_params(axis="both", which="major", labelsize=22)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "value_differences.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def plot_interesting_states(
        self,
        theta_bins=50,
        omega_bins=50,
        first_step_threshold=5.0,
        first_k_sum_threshold=15.0,
        k=5,
    ):
        """Plot total Lyapunov function and highlight states with interesting weight patterns.

        Args:
            theta_bins, omega_bins: Resolution of the state space grid
            first_step_threshold: Threshold for the first step weight
            first_k_sum_threshold: Threshold for sum of first k weights
            k: Number of early steps to consider for sum
        """
        theta = np.linspace(-np.pi * 2, np.pi * 2, theta_bins)
        omega = np.linspace(-8, 8, omega_bins)
        Theta, Omega = np.meshgrid(theta, omega)

        V_total_values = np.zeros_like(Theta)
        small_first_step_states = []  # States with small first step weight
        small_sum_states = []  # States with small sum of first k weights

        for i in range(theta_bins):
            for j in range(omega_bins):
                theta_val = theta[i]
                omega_val = omega[j]

                # Get initial state and value
                obs = np.array([np.cos(theta_val), np.sin(theta_val), omega_val])
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

                # Get weights and total value
                with torch.no_grad():
                    sigma = self.stepnet(obs_tensor)  # Get weights for future values
                    initial_value = float(self.get_value(obs_tensor))
                    initial_residual = self.get_residual_value(obs_tensor)
                    initial_total = np.abs(-initial_value) + float(
                        initial_residual.cpu().numpy().item()
                    )

                # Store the total value
                V_total_values[j, i] = initial_total

                # Analyze weight pattern
                weights = sigma[0].cpu().numpy()
                first_step_weight = weights[-1]
                first_k_sum = np.sum(weights[:k])

                # Classify state based on weight pattern
                if first_step_weight < first_step_threshold:
                    small_first_step_states.append((theta_val, omega_val, weights[:k]))
                if first_k_sum < first_k_sum_threshold:
                    small_sum_states.append((theta_val, omega_val, weights[:k]))

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Plot total Lyapunov function
        contour = ax1.contourf(Theta, Omega, V_total_values, levels=50, cmap="viridis")
        plt.colorbar(contour, ax=ax1)
        ax1.set_xlabel("Theta (radians)")
        ax1.set_ylabel("Theta_dot (rad/s)")
        ax1.set_title("Total Lyapunov Function with Interesting States")

        # Plot states with interesting weight patterns
        if small_first_step_states:
            states = np.array([(s[0], s[1]) for s in small_first_step_states])
            ax1.plot(
                states[:, 0],
                states[:, 1],
                "r.",
                markersize=5,
                alpha=0.7,
                label=f"Small First Step (<{first_step_threshold})",
            )
        if small_sum_states:
            states = np.array([(s[0], s[1]) for s in small_sum_states])
            ax1.plot(
                states[:, 0],
                states[:, 1],
                "g.",
                markersize=5,
                alpha=0.7,
                label=f"Small {k}-Step Sum (<{first_k_sum_threshold})",
            )
        ax1.legend()

        # Modified plotting for weight patterns
        ax2.set_title("Example Weight Patterns (all steps)")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Weight Value")

        # Get full weight patterns for a few example states
        n_examples = min(5, len(small_first_step_states), len(small_sum_states))
        for i in range(n_examples):
            if small_first_step_states:
                # Get the full state and compute all weights
                theta, omega, _ = small_first_step_states[i]
                obs = np.array([np.cos(theta), np.sin(theta), omega])
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    sigma = self.stepnet(obs_tensor)
                weights = sigma[0].cpu().numpy()
                ax2.plot(
                    range(self.n_steps),
                    weights,
                    "r--",
                    alpha=0.7,
                    label="Small First Step" if i == 0 else None,
                )

            if small_sum_states:
                # Get the full state and compute all weights
                theta, omega, _ = small_sum_states[i]
                obs = np.array([np.cos(theta), np.sin(theta), omega])
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    sigma = self.stepnet(obs_tensor)
                weights = sigma[0].cpu().numpy()
                ax2.plot(
                    range(self.n_steps),
                    weights,
                    "g-",
                    alpha=0.7,
                    label="Small Sum" if i == 0 else None,
                )

        ax2.legend()
        ax2.grid(True, alpha=0.3)
        # Add x-axis ticks for every 5 steps
        ax2.set_xticks(range(0, self.n_steps, 5))

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "interesting_states.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def plot_specific_states_weights(self, states_of_interest):
        """Plot weight patterns and Lyapunov values for specific states of interest."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        colors = plt.cm.rainbow(np.linspace(0, 1, len(states_of_interest)))

        for (theta, omega), color in zip(states_of_interest, colors):
            # First collect the full trajectory
            states, _ = self.collect_trajectory([theta, omega], max_steps=self.n_steps + 1)

            # Get weights for the initial state
            obs_tensor = torch.FloatTensor(states[0]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                sigma = self.stepnet(obs_tensor)
            weights = sigma[0].cpu().numpy()

            # Plot weights (σ₁ to σₘ for future states x₁ to xₘ)
            label = f"State (θ={theta:.1f}, ω={omega:.1f})"
            ax1.plot(
                range(1, self.n_steps + 1),
                weights,
                "-",
                color=color,
                alpha=0.8,
                label=label,
                marker="o",
                markersize=4,
            )

            # Get Lyapunov values for all states (x₀ to xₘ)
            lyapunov_values = []
            for state in states[: self.n_steps + 1]:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    V = self.get_residual_value(state_tensor)
                    lyapunov_values.append(V.item())

            # Plot Lyapunov values (V(x₀) to V(xₘ))
            ax2.plot(
                range(len(lyapunov_values)),
                lyapunov_values,
                "-",
                color=color,
                label=label,
                marker="o",
                markersize=4,
                alpha=0.8,
            )

        # Configure weights subplot
        ax1.set_title("Weight Patterns for Specific States", fontsize=14)
        ax1.set_xlabel("Future State Index", fontsize=12)
        ax1.set_ylabel("Weight Value", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(1, self.n_steps + 1, 5))  # Start from 1 for x₁
        ax1.legend()

        # Configure Lyapunov values subplot
        ax2.set_title("Lyapunov Values Along Trajectories", fontsize=14)
        ax2.set_xlabel("State Index", fontsize=12)
        ax2.set_ylabel("Total Lyapunov Value", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(0, self.n_steps + 1, 5))  # Start from 0 for x₀
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "specific_states_analysis.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def plot_lyapunov_values_over_time(self, num_trajectories=10, max_steps=100, seed=42):
        """Plot Lyapunov function values over time for multiple trajectories"""
        np.random.seed(seed)
        torch.manual_seed(seed)

        # First, let's check the value at equilibrium
        equilibrium_tensor = torch.FloatTensor(self.equilibrium_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            V_equilibrium = self.get_residual_value(equilibrium_tensor)
        print("\nDiagnostic Information:")
        print(f"Equilibrium state: {self.equilibrium_state}")
        print(f"Lyapunov value at equilibrium: {V_equilibrium.item():.6f}")

        plt.figure(figsize=(10, 8))
        colors = plt.cm.rainbow(np.linspace(0, 1, num_trajectories))

        for i in range(num_trajectories):
            # Random initial state
            theta = np.random.uniform(-np.pi, np.pi)
            omega = np.random.uniform(-8, 8)

            # Set initial state
            obs = np.array([np.cos(theta), np.sin(theta), omega])
            self.env.reset()
            self.env.unwrapped.state = np.array([theta, omega])

            # Store states and values
            states = [obs]
            lyapunov_values = []

            # Get initial Lyapunov value
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                V_initial = self.get_residual_value(obs_tensor)
            lyapunov_values.append(V_initial.item())

            # Run trajectory
            for step in range(max_steps):
                action, _ = self.rl_model.predict(obs, deterministic=True)
                obs, _, done, _, _ = self.env.step(action)
                states.append(obs)

                # Compute Lyapunov value
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    V = self.get_residual_value(obs_tensor)
                lyapunov_values.append(V.item())

                if done:
                    break

            # Print final value for this trajectory
            print(f"Trajectory {i + 1} final value: {lyapunov_values[-1]:.6f}")

            # Plot trajectory values
            plt.plot(
                lyapunov_values,
                c=colors[i],
                alpha=1.0,
                linewidth=3,
                label=f"Trajectory {i + 1}" if i < 5 else None,
            )

        plt.xlabel("Time Step", fontsize=24)
        plt.ylabel("Generalized Lyapunov Value", fontsize=24)
        plt.tick_params(axis="both", which="major", labelsize=22)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "lyapunov_values_over_time.png"), dpi=300, bbox_inches="tight")
        plt.close()


def main():
    algo_type = "ppo"  # sac or ppo
    # Initialize visualizer with paths to saved models
    n_steps = 15
    
    # Create results directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(
        os.path.dirname(script_dir), "..", "..", "results", "gym_inverted_pendulum", algo_type
    )
    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Figures will be saved to: {results_dir}")
    
    visualizer = StepNetVisualizer(
        model_path=os.path.join(
            os.path.dirname(__file__), "..", "saved_models", algo_type, "pendulum"
        ),
        stepnet_path=os.path.join(
            os.path.dirname(__file__),
            "..",
            "saved_models",
            algo_type,
            f"{n_steps}steps",
            "stepnet_best.pth",
        ),
        residual_path=os.path.join(
            os.path.dirname(__file__),
            "..",
            "saved_models",
            algo_type,
            f"{n_steps}steps",
            "residual_net_best.pth",
        ),
        algo_type=algo_type,
        n_steps=n_steps,
        results_dir=results_dir,
        beta=0.01,  
    )

    # Generate visualizations
    visualizer.plot_lyapunov_values_over_time(num_trajectories=10, max_steps=200, seed=42)
    # visualizer.plot_trajectories_and_values(num_trajectories=20, seed=30)
    visualizer.plot_value_differences(
        theta_bins=100, omega_bins=100, num_trajectories=20, max_steps=200
    )


if __name__ == "__main__":
    main()
