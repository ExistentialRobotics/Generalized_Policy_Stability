import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import os
import sys

# Add the parent directory to path for network imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from network.lyapunov_net import LyapunovNet


class TraditionalLyapunovTrainer:
    def __init__(
        self,
        model_path: str,
        env_id: str = "Pendulum-v1",
        algo_type: str = "ppo",
        hidden_dim: int = 64,
        n_layers: int = 3,
        alpha: float = 0.02,
    ):
        self.algo_type = algo_type.lower()
        if self.algo_type == "ppo":
            self.rl_model = PPO.load(model_path)
            print("Successfully loaded PPO model")
        elif self.algo_type == "sac":
            self.rl_model = SAC.load(model_path)
            print("Successfully loaded SAC model")
        else:
            raise ValueError(f"Unsupported algorithm type: {algo_type}")

        self.env = gym.make(env_id)
        self.state_dim = self.env.observation_space.shape[0]
        self.residual_net = LyapunovNet(
            n_input=self.state_dim, n_hidden=hidden_dim, n_layers=n_layers
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.residual_net.to(self.device)
        self.alpha = alpha

        initial_lr = 5e-4
        self.residual_optimizer = torch.optim.Adam(self.residual_net.parameters(), lr=initial_lr)
        self.residual_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.residual_optimizer, mode="min", factor=0.5, patience=500, verbose=True
        )

        self.equilibrium_state, self.value_at_equilibrium = self.find_practical_equilibrium()
        self.print_equilibrium_info()

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

    def get_value(self, obs: np.ndarray) -> float:
        """Get value from RL model based on algorithm type"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.algo_type == "ppo":
                value = float(self.rl_model.policy.predict_values(obs_tensor).cpu().numpy().item())
            else:  # SAC
                value = float(
                    self.rl_model.critic(
                        obs_tensor, torch.zeros((1, self.env.action_space.shape[0])).to(self.device)
                    )[0]
                    .cpu()
                    .numpy()
                    .item()
                )
        return value

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

    def collect_trajectory(self, initial_state, max_steps=2):
        theta, omega = initial_state
        self.env.reset()
        self.env.unwrapped.state = np.array([theta, omega])
        obs = np.array([np.cos(theta), np.sin(theta), omega])
        states = []
        for _ in range(max_steps):
            states.append(obs)
            action, _ = self.rl_model.predict(obs, deterministic=True)
            next_obs, _, done, _, _ = self.env.step(action)
            obs = next_obs
            if done:
                break
        return np.array(states)

    def get_total_lyapunov(self, state):
        """
        Compute the Lyapunov function value:
        V(x) = |V_RL(x) - V_RL(x*)| + ‖φ(x) - φ(x*)‖² + α‖x - x*‖²
        """
        with torch.no_grad():
            if self.algo_type == "ppo":
                v_rl = self.rl_model.policy.predict_values(state)
            else:  # SAC
                action = self.rl_model.actor(state)[0]
                q1, q2 = self.rl_model.critic(state, action.unsqueeze(0))
                v_rl = torch.min(q1, q2)

        phi_x = self.residual_net(state)
        equilibrium_tensor = torch.FloatTensor(self.equilibrium_state).to(self.device)
        phi_equilibrium = self.residual_net(equilibrium_tensor)

        term1 = torch.abs(v_rl - self.value_at_equilibrium)
        term2 = torch.pow(torch.norm(phi_x - phi_equilibrium), 2)
        state_diff = state - equilibrium_tensor
        term3 = self.alpha * torch.pow(torch.norm(state_diff), 2)

        V = term1 + term2 + term3
        return V

    def compute_loss(self, initial_states):
        loss = 0.0
        for init_state in initial_states:
            states = self.collect_trajectory(init_state, max_steps=2)
            if len(states) < 2:
                continue
            x_k = torch.FloatTensor(states[0]).unsqueeze(0).to(self.device)
            x_k1 = torch.FloatTensor(states[1]).unsqueeze(0).to(self.device)
            V_k = self.get_total_lyapunov(x_k)
            V_k1 = self.get_total_lyapunov(x_k1)
            violation = torch.relu(V_k1 - V_k + self.alpha * V_k)
            loss += violation
        avg_loss = loss / len(initial_states)
        return avg_loss

    def train(self, n_epochs=1000, batch_size=256):
        save_dir = os.path.join(
            os.path.dirname(__file__), "..", "saved_models", self.algo_type, "traditional"
        )
        os.makedirs(save_dir, exist_ok=True)
        print(f"Models will be saved to: {save_dir}")

        losses = []
        best_loss = float("inf")
        best_residual_state = None
        max_grad_norm = 5.0
        total_start_time = time.time()

        for epoch in range(n_epochs):
            epoch_start_time = time.time()
            initial_states = []
            for _ in range(batch_size):
                theta = np.random.uniform(-np.pi, np.pi)
                omega = np.random.uniform(-8, 8)
                initial_states.append(np.array([theta, omega]))

            self.residual_optimizer.zero_grad()
            loss = self.compute_loss(initial_states)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.residual_net.parameters(), max_grad_norm)
            self.residual_optimizer.step()
            self.residual_scheduler.step(loss)
            losses.append(loss.item())

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_residual_state = self.residual_net.state_dict()
                print(f"New best model found at epoch {epoch + 1} with loss: {best_loss:.4f}")
                torch.save(best_residual_state, f"{save_dir}/residual_net_best.pth")

            if (epoch + 1) % 1 == 0:
                print(
                    f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}, Time: {time.time() - epoch_start_time:.2f}s"
                )

        print(f"\nTraining completed in {(time.time() - total_start_time) / 60:.2f} minutes")
        print(f"Best loss achieved: {best_loss:.4f}")

        torch.save(self.residual_net.state_dict(), f"{save_dir}/residual_net_final.pth")

        plt.figure(figsize=(10, 5))
        plt.plot(losses, label="Stability Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Traditional Lyapunov Training Loss ({self.algo_type})")
        plt.legend()
        plt.savefig(f"{save_dir}/training_losses.png")
        plt.close()

        return losses


def main():
    algorithms = ["ppo", "sac"]
    n_epochs = 5000
    batch_size = 256

    for algo in algorithms:
        print(f"\n{'=' * 50}")
        print(f"Traditional Lyapunov Training with {algo.upper()} algorithm")
        print(f"{'=' * 50}")

        try:
            trainer = TraditionalLyapunovTrainer(
                model_path=os.path.join(
                    os.path.dirname(__file__), "..", "saved_models", algo, "pendulum"
                ),
                algo_type=algo,
            )
            trainer.train(n_epochs=n_epochs, batch_size=batch_size)
            print(f"Successfully completed traditional LF training for {algo.upper()}")
        except Exception as e:
            print(f"Error during training {algo.upper()}: {str(e)}")
            continue

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
