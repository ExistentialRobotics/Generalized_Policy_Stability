import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

# Add the parent directory to path
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from network.PolicyNet import StepNet
from network.lyapunov_net import LyapunovNet


class StepNetTrainer:
    def __init__(
        self,
        model_path: str,
        env_id: str = "Pendulum-v1",
        algo_type: str = "ppo",
        n_steps: int = 20,
        hidden_dim: int = 64,
        n_layers: int = 3,
        alpha: float = 0.05,
        beta: float = 0.01,
    ):
        """
        Initialize trainer for finding Lyapunov function using any RL policy

        Args:
            model_path: Path to saved RL model
            env_id: Gymnasium environment ID
            algo_type: RL algorithm type ('ppo' or 'sac')
            n_steps: Number of future steps to consider
            hidden_dim: Hidden dimension for neural networks
            n_layers: Number of layers for neural networks
            alpha: Decay parameter for stability condition
            beta: Weight parameter for quadratic term in Lyapunov construction
        """
        # Load trained RL model
        self.algo_type = algo_type.lower()

        try:
            if self.algo_type == "ppo":
                self.rl_model = PPO.load(model_path)
                print("Successfully loaded PPO model")
            elif self.algo_type == "sac":
                self.rl_model = SAC.load(model_path)
                print("Successfully loaded SAC model")
            else:
                raise ValueError(f"Unsupported algorithm type: {algo_type}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

        self.env = gym.make(env_id)
        self.state_dim = self.env.observation_space.shape[0]

        # Initialize networks
        self.stepnet = StepNet(
            n_input=self.state_dim, n_hidden=hidden_dim, n_steps=n_steps, n_layers=n_layers
        )
        self.residual_net = LyapunovNet(
            n_input=self.state_dim, n_hidden=hidden_dim, n_layers=n_layers
        )

        self.n_steps = n_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stepnet.to(self.device)
        self.residual_net.to(self.device)

        # Initialize optimizers and schedulers
        initial_lr = 2e-4
        self.setup_optimizers(initial_lr)
        self.alpha = alpha
        self.beta = beta

        # Find practical equilibrium state
        self.equilibrium_state, self.value_at_equilibrium = self.find_practical_equilibrium()
        self.print_equilibrium_info()

    def get_value(self, obs: np.ndarray) -> float:
        """Get value from RL model based on algorithm type"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.algo_type == "ppo":
                value = float(self.rl_model.policy.predict_values(obs_tensor).cpu().numpy().item())
            else:  # SAC
                # For SAC, we use the Q-value with zero action as an approximation
                value = float(
                    self.rl_model.critic(
                        obs_tensor, torch.zeros((1, self.env.action_space.shape[0])).to(self.device)
                    )[0]
                    .cpu()
                    .numpy()
                    .item()
                )
        return value

    def setup_optimizers(self, initial_lr: float):
        """Setup optimizers and learning rate schedulers"""
        self.optimizer = torch.optim.Adam(self.stepnet.parameters(), lr=initial_lr)
        self.residual_optimizer = torch.optim.Adam(self.residual_net.parameters(), lr=initial_lr)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=500, verbose=True
        )
        self.residual_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.residual_optimizer, mode="min", factor=0.5, patience=500, verbose=True
        )

    def print_equilibrium_info(self):
        """Print information about found equilibrium state"""
        print("\nPractical equilibrium found:")
        if self.state_dim == 3:  # Special case for pendulum
            print(
                f"State: theta={np.arctan2(self.equilibrium_state[1], self.equilibrium_state[0]):.6f} rad, "
                f"omega={self.equilibrium_state[2]:.6f} rad/s"
            )
        else:
            print(f"State: {self.equilibrium_state}")
        print(f"Value at equilibrium: {self.value_at_equilibrium:.6f}")

    def find_practical_equilibrium(self, simulation_steps: int = 200) -> tuple:
        """Find practical equilibrium by simulating system"""
        self.env.reset()
        if hasattr(self.env.unwrapped, "state"):
            self.env.unwrapped.state = np.zeros_like(self.env.unwrapped.state)
        obs, _ = self.env.reset()

        states = []
        values = []

        for _ in range(simulation_steps):
            states.append(obs)
            value = self.get_value(obs)
            action, _ = self.rl_model.predict(obs, deterministic=True)
            values.append(value)
            obs, _, _, _, _ = self.env.step(action)

        last_n = 10
        equilibrium_state = np.mean(states[-last_n:], axis=0)
        equilibrium_value = np.abs(-np.mean(values[-last_n:]))

        return equilibrium_state, equilibrium_value

    def collect_trajectory(self, initial_state, max_steps=200):
        """Collect trajectory and corresponding values using RL policy (batched RL predictions)"""
        theta, omega = initial_state

        # Reset and set state using unwrapped env (correct format)
        self.env.reset()
        self.env.unwrapped.state = np.array([theta, omega])

        # First observation
        obs = np.array([np.cos(theta), np.sin(theta), omega])

        # Collect all states first by stepping through environment
        states = []
        actions_list = []
        
        for _ in range(max_steps):
            states.append(obs.copy())
            
            # Get action (still need to do this sequentially for stepping)
            action, _ = self.rl_model.predict(obs, deterministic=True)
            actions_list.append(action)
            
            # Step environment
            next_obs, _, done, _, _ = self.env.step(action)
            obs = next_obs

            if done:
                break

        if len(states) == 0:
            return np.array([]), np.array([])

        # Batch compute all values at once
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        values = []

        with torch.no_grad():
            if self.algo_type == "ppo":
                # Batch predict values for all states
                values_tensor = self.rl_model.policy.predict_values(states_tensor)
                values = np.abs(-values_tensor.cpu().numpy().flatten())
            else:  # SAC
                # For SAC, batch compute Q-values with corresponding actions
                actions_tensor = torch.FloatTensor(np.array(actions_list)).to(self.device)
                q1, q2 = self.rl_model.critic(states_tensor, actions_tensor)
                values_tensor = torch.min(q1, q2)
                values = np.abs(-values_tensor.cpu().numpy().flatten())

        return np.array(states), np.array(values)

    def get_residual_value(self, state):
        """
        Compute the Lyapunov function value with the corrected construction:
        V(x) = |V_RL(x) - V_RL(x*)| + ‖φ(x) - φ(x*)‖² + β‖x - x*‖²
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

        # Total Lyapunov value
        V = term1 + term2 + term3

        return V

    def compute_loss(self, initial_states):
        """
        Compute loss with only stability condition
        """
        stability_loss = 0.0

        for init_state in initial_states:
            # Collect trajectory and compute Lyapunov values
            states, _ = self.collect_trajectory(init_state, max_steps=self.n_steps + 1)

            if len(states) < self.n_steps:
                continue

            init_obs = np.array([np.cos(init_state[0]), np.sin(init_state[0]), init_state[1]])
            init_state_tensor = torch.FloatTensor(init_obs).to(self.device)

            # Get weights from StepNet
            sigma = self.stepnet(init_state_tensor)

            # Compute all Lyapunov values at once for the trajectory
            trajectory_states = torch.FloatTensor(states[: self.n_steps + 1]).to(self.device)
            V_values = torch.zeros(self.n_steps + 1).to(self.device)

            for i, state in enumerate(trajectory_states):
                V_values[i] = self.get_residual_value(state.unsqueeze(0))

            # Initial value for stability condition
            V_0 = V_values[0]

            # Future values for stability condition
            future_values = V_values[1 : self.n_steps + 1]

            # Stability condition
            weighted_sum = torch.sum(sigma[0, : self.n_steps] * future_values) / self.n_steps
            stability_violation = torch.relu(weighted_sum - V_0 + self.alpha * V_0)
            stability_loss += stability_violation

        avg_stability_loss = stability_loss / len(initial_states)
        total_loss = avg_stability_loss

        return (
            total_loss,
            avg_stability_loss,
            torch.tensor(0.0).to(self.device),
        )  # Return 0 for positivity loss

    def train(self, n_epochs=1000, batch_size=256):
        """
        Train the networks using only stability loss
        """
        print("\nStarting training with new Lyapunov function construction...")

        # Create directory for saving models if it doesn't exist
        # Save in the same location as RL models for consistency
        base_dir = os.path.join(os.path.dirname(__file__), "..", "saved_models")
        save_dir = os.path.join(base_dir, self.algo_type, f"{self.n_steps}steps")
        save_dir = os.path.abspath(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        print(f"Models will be saved to: {save_dir}")

        losses = []
        stability_losses = []

        # Track best model
        best_loss = float("inf")
        best_stepnet_state = None
        best_residual_state = None

        max_grad_norm = 5.0
        total_start_time = time.time()

        for epoch in range(n_epochs):
            epoch_start_time = time.time()

            # Sample random initial states
            initial_states = []
            for _ in range(batch_size):
                theta = np.random.uniform(-np.pi, np.pi)
                omega = np.random.uniform(-8, 8)
                initial_states.append(np.array([theta, omega]))

            # Compute and optimize loss
            self.optimizer.zero_grad()
            self.residual_optimizer.zero_grad()
            total_loss, stability_loss, _ = self.compute_loss(initial_states)
            total_loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.stepnet.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.residual_net.parameters(), max_grad_norm)

            self.optimizer.step()
            self.residual_optimizer.step()

            self.scheduler.step(total_loss)
            self.residual_scheduler.step(total_loss)

            # Store losses
            losses.append(total_loss.item())
            stability_losses.append(stability_loss.item())

            # Save best model
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_stepnet_state = self.stepnet.state_dict()
                best_residual_state = self.residual_net.state_dict()
                print(f"New best model found at epoch {epoch + 1} with loss: {best_loss:.4f}")

                torch.save(best_stepnet_state, f"{save_dir}/stepnet_best.pth")
                torch.save(best_residual_state, f"{save_dir}/residual_net_best.pth")

            # Print progress
            if (epoch + 1) % 1 == 0:
                print(
                    f"Epoch {epoch + 1}/{n_epochs}, "
                    f"Stability Loss: {stability_loss.item():.4f}, "
                    f"Time: {time.time() - epoch_start_time:.2f}s"
                )

        print(f"\nTraining completed in {(time.time() - total_start_time) / 60:.2f} minutes")
        print(f"Best loss achieved: {best_loss:.4f}")

        # Save final models
        torch.save(self.stepnet.state_dict(), f"{save_dir}/stepnet_final.pth")
        torch.save(self.residual_net.state_dict(), f"{save_dir}/residual_net_final.pth")

        # Plot loss
        plt.figure(figsize=(10, 5))
        plt.plot(stability_losses, label="Stability Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training Loss ({self.algo_type}, {self.n_steps} steps)")
        plt.legend()
        plt.savefig(f"{save_dir}/training_losses.png")
        plt.close()

        return losses, stability_losses, [0] * len(losses)  # Return zeros for positivity losses


def main():
    # Configuration
    step_sizes = [15]
    algorithms = ["ppo", "sac"]
    n_epochs = 1000
    batch_size = 512

    for algo in algorithms:
        print(f"\n{'=' * 50}")
        print(f"Training with {algo.upper()} algorithm")
        print(f"{'=' * 50}")

        for n_steps in step_sizes:
            print(f"\n{'-' * 30}")
            print(f"Training for {n_steps} steps")
            print(f"{'-' * 30}")

            try:
                # Initialize trainer
                trainer = StepNetTrainer(
                    model_path=os.path.join(
                        os.path.dirname(__file__), "..", "saved_models", algo, "pendulum"
                    ),
                    algo_type=algo,
                    n_steps=n_steps,
                )

                # Train networks
                trainer.train(n_epochs=n_epochs, batch_size=batch_size)

                print(f"Successfully completed training for {algo.upper()} with {n_steps} steps")

            except Exception as e:
                print(f"Error during training {algo.upper()} with {n_steps} steps: {str(e)}")
                continue

            # Optional: Clear GPU memory between runs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
