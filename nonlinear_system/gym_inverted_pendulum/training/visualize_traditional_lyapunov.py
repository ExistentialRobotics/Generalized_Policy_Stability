import torch
import numpy as np
import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from network.lyapunov_net import LyapunovNet
from stable_baselines3 import PPO, SAC


def get_value(rl_model, obs_tensor, algo_type, device, env):
    with torch.no_grad():
        if algo_type == "ppo":
            value = float(rl_model.policy.predict_values(obs_tensor).cpu().numpy().item())
        else:  # SAC
            value = float(
                rl_model.critic(obs_tensor, torch.zeros((1, env.action_space.shape[0])).to(device))[
                    0
                ]
                .cpu()
                .numpy()
                .item()
            )
    return value


def get_total_lyapunov(
    rl_model,
    residual_net,
    obs_tensor,
    equilibrium_tensor,
    value_at_equilibrium,
    algo_type,
    alpha,
    device,
):
    with torch.no_grad():
        if algo_type == "ppo":
            v_rl = rl_model.policy.predict_values(obs_tensor)
        else:  # SAC
            action = rl_model.actor(obs_tensor)[0]
            q1, q2 = rl_model.critic(obs_tensor, action.unsqueeze(0))
            v_rl = torch.min(q1, q2)
    phi_x = residual_net(obs_tensor)
    phi_equilibrium = residual_net(equilibrium_tensor)
    term1 = torch.abs(v_rl - value_at_equilibrium)
    term2 = torch.pow(torch.norm(phi_x - phi_equilibrium), 2)
    state_diff = obs_tensor - equilibrium_tensor
    term3 = alpha * torch.pow(torch.norm(state_diff), 2)
    V = term1 + term2 + term3
    return V.item()


def find_practical_equilibrium(env, rl_model, algo_type, device):
    env.reset()
    if hasattr(env.unwrapped, "state"):
        env.unwrapped.state = np.zeros_like(env.unwrapped.state)
    obs, _ = env.reset()
    states = []
    values = []
    for _ in range(200):
        states.append(obs)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        value = get_value(rl_model, obs_tensor, algo_type, device, env)
        action, _ = rl_model.predict(obs, deterministic=True)
        values.append(value)
        obs, _, _, _, _ = env.step(action)
    last_n = 10
    equilibrium_state = np.mean(states[-last_n:], axis=0)
    equilibrium_value = np.abs(-np.mean(values[-last_n:]))
    return equilibrium_state, equilibrium_value


def main():
    algo_type = "ppo"  # or 'sac'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    alpha = 0.02

    # Paths
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "saved_models", algo_type, "pendulum"
    )
    residual_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "saved_models",
        algo_type,
        "traditional",
        "residual_net_best.pth",
    )

    # Load RL model and residual net
    if algo_type == "ppo":
        rl_model = PPO.load(model_path)
    else:
        rl_model = SAC.load(model_path)
    residual_net = LyapunovNet(n_input=state_dim, n_hidden=64, n_layers=3)
    residual_net.load_state_dict(torch.load(residual_path, map_location=device))
    residual_net.to(device)
    residual_net.eval()

    # Find equilibrium state and value
    equilibrium_state, value_at_equilibrium = find_practical_equilibrium(
        env, rl_model, algo_type, device
    )
    equilibrium_tensor = torch.FloatTensor(equilibrium_state).unsqueeze(0).to(device)
    print(f"Equilibrium state: {equilibrium_state}")
    print(f"Value at equilibrium: {value_at_equilibrium}")

    # State grid
    theta_vals = np.linspace(-np.pi, np.pi, 100)
    omega_vals = np.linspace(-8, 8, 100)
    Theta, Omega = np.meshgrid(theta_vals, omega_vals)
    V_vals = np.zeros_like(Theta)
    DeltaV_vals = np.zeros_like(Theta)

    for i in range(Theta.shape[0]):
        for j in range(Theta.shape[1]):
            theta = Theta[i, j]
            omega = Omega[i, j]
            obs = np.array([np.cos(theta), np.sin(theta), omega])
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

            # V(x_k)
            V_k = get_total_lyapunov(
                rl_model,
                residual_net,
                obs_tensor,
                equilibrium_tensor,
                value_at_equilibrium,
                algo_type,
                alpha,
                device,
            )

            # Simulate one step to get x_{k+1}
            env.reset()
            env.unwrapped.state = np.array([theta, omega])
            next_obs, _ = env.reset()
            action, _ = rl_model.predict(obs, deterministic=True)
            next_obs, _, _, _, _ = env.step(action)
            next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(device)
            V_k1 = get_total_lyapunov(
                rl_model,
                residual_net,
                next_obs_tensor,
                equilibrium_tensor,
                value_at_equilibrium,
                algo_type,
                alpha,
                device,
            )

            # Lyapunov decrease condition
            delta_V = V_k1 - V_k

            V_vals[i, j] = V_k
            DeltaV_vals[i, j] = delta_V

    # Plot V(x)
    plt.figure(figsize=(10, 5))
    plt.contourf(Theta, Omega, V_vals, levels=50, cmap="viridis")
    plt.colorbar()
    plt.xlabel("Theta (rad)")
    plt.ylabel("Omega (rad/s)")
    plt.title("Traditional Lyapunov Function Value $V(x)$")
    plt.savefig("traditional_lyapunov_value.png", bbox_inches="tight", dpi=200)
    plt.close()

    # Plot Lyapunov decrease condition
    plt.figure(figsize=(10, 5))
    plt.contourf(Theta, Omega, DeltaV_vals, levels=50, cmap="coolwarm")
    plt.colorbar()
    plt.xlabel("Theta (rad)")
    plt.ylabel("Omega (rad/s)")
    plt.title(r"Lyapunov Decrease Condition $\Delta V(x) = V(x_{k+1}) - V(x_k) + \alpha V(x_k)$")
    # Overlay violation points
    violation_mask = DeltaV_vals > 0
    plt.scatter(Theta[violation_mask], Omega[violation_mask], color="red", s=2, label="Violation")
    plt.legend()
    plt.savefig("traditional_lyapunov_decrease.png", bbox_inches="tight", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
