#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: kehan
"""

import numpy as np
import torch

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt


import gymnasium as gym


def evaluate_rl_controller(model, theta_bins=50, omega_bins=50, N=10, num_trajectories=1):
    # Create a grid of theta and omega values
    theta = np.linspace(-np.pi * 2, np.pi * 2, theta_bins)
    omega = np.linspace(-8, 8, omega_bins)
    Theta, Omega = np.meshgrid(theta, omega)

    sin_theta = np.sin(Theta.ravel())
    cos_theta = np.cos(Theta.ravel())
    Omega_flat = Omega.ravel()

    V_values = np.zeros_like(Theta.ravel())
    U_values = np.zeros_like(Theta.ravel())
    V_diff_values = np.zeros_like(Theta.ravel())

    # Create a dummy environment for simulation
    env = gym.make("Pendulum-v1")

    # Store all trajectories
    all_trajectories = []

    for i in range(len(Theta.ravel())):
        # Initial state
        obs = np.array([cos_theta[i], sin_theta[i], Omega_flat[i]])
        action, _states = model.predict(obs, deterministic=True)

        if isinstance(model, SAC):
            obs_tensor = torch.tensor(obs[None, :], dtype=torch.float32).to(model.device)
            # Make sure action is 2D tensor with shape [batch_size, action_dim]
            action_tensor = torch.tensor(action[None, :], dtype=torch.float32).to(model.device)
            with torch.no_grad():
                # SAC critic returns tuple of (q1, q2)
                q1, q2 = model.critic(obs_tensor, action_tensor)
                # Take minimum of both Q-values as per SAC design
                V_value = -min(q1.item(), q2.item())  # Negate for Lyapunov-like function
        elif isinstance(model, PPO):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(model.device)
            with torch.no_grad():
                # PPO directly outputs value function
                V_value = -float(
                    model.policy.predict_values(obs_tensor).cpu().numpy()
                )  # Negate for Lyapunov-like function

        V_values[i] = V_value
        U_values[i] = float(action.item() if hasattr(action, "item") else action[0])

        # Store trajectory points
        trajectory_points = []
        current_obs = obs.copy()
        trajectory_points.append([np.arctan2(current_obs[1], current_obs[0]), current_obs[2]])

        # Reset environment before simulating trajectory
        env.reset()
        env.state = np.array([np.arctan2(current_obs[1], current_obs[0]), current_obs[2]])

        for step in range(N):
            action, _ = model.predict(current_obs, deterministic=True)
            next_obs, _, _, _, _ = env.step(action)
            trajectory_points.append([np.arctan2(next_obs[1], next_obs[0]), next_obs[2]])
            current_obs = next_obs

        all_trajectories.append(trajectory_points)

        # Compute value for future states
        future_values = []
        for point in trajectory_points[1:]:  # Skip initial state
            theta = point[0]
            omega = point[1]
            obs_future = np.array([np.cos(theta), np.sin(theta), omega])
            try:
                if isinstance(model, SAC):
                    obs_tensor = torch.tensor(obs_future[None, :], dtype=torch.float32).to(
                        model.device
                    )
                    action_tensor = torch.tensor(action[None, :], dtype=torch.float32).to(
                        model.device
                    )
                    with torch.no_grad():
                        q_values = model.critic(obs_tensor, action_tensor)
                        V_future = -float(min(q_values).cpu().numpy())
                elif isinstance(model, PPO):
                    obs_tensor = (
                        torch.tensor(obs_future, dtype=torch.float32).unsqueeze(0).to(model.device)
                    )
                    with torch.no_grad():
                        V_future = -float(model.policy.predict_values(obs_tensor).cpu().numpy())
                future_values.append(V_future)
            except Exception as e:
                print(f"Error computing future value: {e}")
                future_values.append(0.0)

        # Compute average future value minus initial value
        avg_future_value = float(np.mean(future_values)) if future_values else 0.0
        V_diff_values[i] = avg_future_value - V_value

    # Reshape arrays for plotting
    V_values = V_values.reshape(Theta.shape)
    U_values = U_values.reshape(Theta.shape)
    V_diff_values = V_diff_values.reshape(Theta.shape)

    model_name = "SAC" if isinstance(model, SAC) else "PPO"

    plt.figure(figsize=(20, 6))

    # Plot 1: Value function with trajectories
    plt.subplot(1, 3, 1)
    contour = plt.contourf(Theta, Omega, V_values, levels=50, cmap="viridis")
    plt.colorbar(contour)

    # Randomly select and plot trajectories
    np.random.seed(42)  # for reproducibility
    # selected_indices = np.random.choice(len(all_trajectories), num_trajectories, replace=False)

    # for idx in selected_indices:
    #     traj = np.array(all_trajectories[idx])
    #     # Plot trajectory
    #     plt.plot(traj[:, 0], traj[:, 1], 'w-', linewidth=1, alpha=0.7)

    #     # Plot start point (green) with larger marker
    #     plt.plot(traj[0, 0], traj[0, 1], 'go', markersize=8, label='Start' if idx == selected_indices[0] else "")

    #     # Plot end point (red) with larger marker
    #     plt.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=8, label='End' if idx == selected_indices[0] else "")

    #     # Add arrow to show direction
    #     arrow_idx = len(traj) // 2  # Place arrow in middle of trajectory
    #     if arrow_idx > 0:  # Only add arrow if trajectory has multiple points
    #         plt.arrow(traj[arrow_idx, 0], traj[arrow_idx, 1],
    #                  (traj[arrow_idx+1, 0] - traj[arrow_idx, 0]) * 2,
    #                  (traj[arrow_idx+1, 1] - traj[arrow_idx, 1]) * 2,
    #                  head_width=0.1, head_length=0.2, fc='w', ec='w', alpha=0.7)

    plt.xlabel("Theta (radians)", fontsize=16)
    plt.ylabel("Omega (rad/s)", fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(f"{model_name} Value Function", fontsize=18)
    plt.legend(fontsize=12)

    # Plot 2: Control policy
    plt.subplot(1, 3, 2)
    contour = plt.contourf(Theta, Omega, U_values, levels=50, cmap="viridis")
    plt.colorbar(contour)
    plt.xlabel("Theta (radians)", fontsize=16)
    plt.ylabel("Omega (rad/s)", fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(f"{model_name} Control Policy", fontsize=18)

    # Plot 3: Value difference (average future - initial)
    plt.subplot(1, 3, 3)
    contour = plt.contourf(Theta, Omega, V_diff_values, levels=50, cmap="RdBu", center=0)
    plt.colorbar(contour)
    plt.xlabel("Theta (radians)", fontsize=16)
    plt.ylabel("Omega (rad/s)", fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(f"Average Future Value - Initial Value\n(N={N} steps)", fontsize=18)

    plt.suptitle(f"{model_name} Evaluation", fontsize=18, y=1.05)
    plt.tight_layout()
    plt.savefig(f"{model_name}_Evaluation.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_actions(actions1, actions2, title, filename, y_axis_name="Control Input"):
    plt.figure(figsize=(10, 6))

    # Use different line styles for the two plots
    # plt.plot(actions1, label='SAC Controller', linestyle='-', linewidth=3)
    plt.plot(actions2, label="PPO Controller", linestyle="-", linewidth=4)

    # Increase font sizes
    plt.title(title, fontsize=24)
    plt.xlabel("Time Step", fontsize=20)
    plt.ylabel(y_axis_name, fontsize=20)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Enlarge the legend
    plt.legend(fontsize=18)

    # plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.close()  # Close instead of show to avoid display issues


def train_model(env_name, time_steps=1000000, algo="ppo"):
    """
    Train an RL model.

    Args:
    env_name: The name of the environment ('pendulum' or 'cartpole').
    time_steps: Number of training steps.
    algo: Algorithm to use ('ppo' or 'sac')

    Returns:
    The trained model.
    """
    print(f"Creating environment from the given name '{env_name}'")

    # Map environment names to gym environment IDs
    env_mapping = {"pendulum": "Pendulum-v1", "cartpole": "CartPole-v1"}

    gym_env_id = env_mapping.get(env_name)
    if not gym_env_id:
        raise ValueError(f"Unknown environment: {env_name}")

    # Create the environment
    env = make_vec_env(gym_env_id, n_envs=1)

    # Create and train the model based on the algorithm
    if algo.lower() == "ppo":
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=time_steps)
    elif algo.lower() == "sac":
        model = SAC("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=500000)  # sac usually uses fewer timesteps
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    return model


def test_model(model, env, num_steps=200):
    """
    Test the trained model and collect control actions and value function data.

    Args:
    model: The trained model.
    env: The testing environment.
    num_steps: Number of steps to test the model.

    Returns:
    A tuple of lists containing actions and value function data.
    """
    obs, _ = env.reset()
    actions = []
    value_function = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)

        # Only render if render_mode is not None
        if hasattr(env, "render_mode") and env.render_mode is not None:
            env.render()

        # Store action (handle both scalar and array actions)
        if isinstance(action, np.ndarray):
            actions.append(action[0])
        else:
            actions.append(action)

        print(f"control input: {action}")

        try:
            # Evaluate the value function
            if isinstance(model, SAC):
                # For SAC, we need to handle the critic differently
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = model.critic(obs_tensor, action_tensor)
                    value = q_values[0].cpu().numpy().min()
            elif isinstance(model, PPO):
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    value = float(model.policy.predict_values(obs_tensor).cpu().numpy())

            value_function.append(value)
        except Exception as e:
            print(f"Error computing value function: {e}")
            value_function.append(0)  # Default value

        if done:
            obs, _ = env.reset()

    return actions, value_function


def plot_trajectories_with_values(
    model, initial_states=None, num_trajectories=5, num_steps=200, seed=42
):
    """
    Plot trajectories from specified initial states and their corresponding value function values.

    Args:
        model: The trained RL model (PPO or SAC)
        initial_states: List of tuples (theta, omega) for initial states. If None, uses random states.
        num_trajectories: Number of trajectories to plot (only used if initial_states is None)
        num_steps: Number of steps to simulate for each trajectory
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("Pendulum-v1", render_mode="human")
    env.reset(seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use provided initial states or generate random ones
    if initial_states is None:
        initial_states = [
            (np.random.uniform(-np.pi, np.pi), np.random.uniform(-8, 8))
            for _ in range(num_trajectories)
        ]

    all_trajectories = []
    all_values = []

    for theta, omega in initial_states:
        # Set initial state
        obs = np.array([np.cos(theta), np.sin(theta), omega])
        env.reset()
        env.unwrapped.state = np.array([theta, omega])  # Directly set the state

        trajectory = [[theta, omega]]
        values = []

        # Get initial value
        if isinstance(model, SAC):
            obs_tensor = torch.tensor(obs[None, :], dtype=torch.float32).to(device)
            with torch.no_grad():
                action = model.actor(obs_tensor)[0]
                action = action.unsqueeze(0)
                q1, q2 = model.critic(obs_tensor, action)
                value = -float(torch.min(q1, q2).cpu().numpy().item())
        else:  # PPO
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                value = -float(model.policy.predict_values(obs_tensor).cpu().numpy().item())
        values.append(value)

        # Run trajectory
        current_theta = theta
        for _ in range(num_steps):
            action, _ = model.predict(obs, deterministic=True)
            print(f"control input: {action}")
            obs, _, done, _, _ = env.step(action)

            # Record state and value
            new_theta = np.arctan2(obs[1], obs[0])
            # Adjust theta for continuity
            if new_theta - current_theta > np.pi:
                new_theta -= 2 * np.pi
            elif new_theta - current_theta < -np.pi:
                new_theta += 2 * np.pi
            current_theta = new_theta

            omega = obs[2]
            trajectory.append([current_theta, omega])

            # Compute value for this state
            if isinstance(model, SAC):
                obs_tensor = torch.tensor(obs[None, :], dtype=torch.float32).to(device)
                with torch.no_grad():
                    action = model.actor(obs_tensor)[0]
                    action = action.unsqueeze(0)
                    q1, q2 = model.critic(obs_tensor, action)
                    value = -float(torch.min(q1, q2).cpu().numpy().item())
            else:  # PPO
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    value = -float(model.policy.predict_values(obs_tensor).cpu().numpy().item())
            values.append(value)

            if done:
                break

        all_trajectories.append(np.array(trajectory))
        all_values.append(np.array(values))

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot trajectories
    ax1.set_title("State Space Trajectories", fontsize=14)
    ax1.set_xlabel("Theta (radians)", fontsize=12)
    ax1.set_ylabel("Omega (rad/s)", fontsize=12)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(initial_states)))
    for traj, color, (init_theta, init_omega) in zip(all_trajectories, colors, initial_states):
        ax1.plot(
            traj[:, 0],
            traj[:, 1],
            c=color,
            alpha=0.7,
            label=f"Initial (θ={init_theta:.1f}, ω={init_omega:.1f})",
        )
        ax1.plot(traj[0, 0], traj[0, 1], "go", markersize=8)  # Start point
        ax1.plot(traj[-1, 0], traj[-1, 1], "ro", markersize=8)  # End point

    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot value functions
    ax2.set_title("Value Function Evolution", fontsize=14)
    ax2.set_xlabel("Time Step", fontsize=12)
    ax2.set_ylabel("Value", fontsize=12)

    for values, color, (init_theta, init_omega) in zip(all_values, colors, initial_states):
        ax2.plot(values, c=color, alpha=0.7)

    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("trajectories_and_values.png", dpi=300, bbox_inches="tight")
    plt.close()

    env.close()


def main():
    # Training

    algo = "ppo"

    # for inverted_pendulum
    # train_env = make_vec_env('Pendulum-v1', n_envs = 1)
    # train_env = make_vec_env(lambda: PendulumEnv(), n_envs=1)

    # for cart-pole
    # train_env = make_vec_env('CartPole-v1', n_envs=1)
    # train_env = make_vec_env(lambda: CartPoleEnv(), n_envs=1)

    # model = train_model('pendulum', algo=algo)
    # model.save(os.path.join(os.path.dirname(__file__), "..", "saved_models", algo, "pendulum"))

    # Testing/Visualization
    # test_env = gym.make("Pendulum-v1", render_mode="human")

    import os

    # Load both models
    if algo == "sac":
        model = SAC.load(
            os.path.join(os.path.dirname(__file__), "..", "saved_models", algo, "pendulum")
        )
    else:
        model = PPO.load(
            os.path.join(os.path.dirname(__file__), "..", "saved_models", algo, "pendulum")
        )

    # Evaluate both models
    # evaluate_rl_controller(model_sac)  # This creates SAC_Evaluation.png
    # evaluate_rl_controller(model_sac, N=30)  # This creates PPO_Evaluation.png

    # Get trajectories and values for both models
    # sac_actions, sac_values = test_model(model_ppo, test_env)
    # ppo_actions, ppo_values = test_model(model_ppo, test_env)

    # # Plot comparisons
    # plot_actions(sac_actions, ppo_actions, "Control Actions Comparison", "control_actions_comparison.png")
    # plot_actions(sac_values, ppo_values, "Value Function Comparison", "value_function_comparison.png", y_axis_name="Value")

    # with open('rl_values_cartpole.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['sac_values', 'ppo_values'])
    #     for sac_val, ppo_val in zip(sac_values, ppo_values):
    #         writer.writerow([sac_val, ppo_val])

    # Run the visualization
    # actions, values = test_model(model_sac, test_env, num_steps=200)  # Will run for 200 steps

    # test_env.close()  # Clean up the environment

    # Specify initial states to test
    initial_states = [
        (0, 2),  # Upright with positive velocity
        (0, -1),  # Upright with negative velocity
        (np.pi, 0),  # Downward position
        (0, -7),  # Upright with larger negative velocity
    ]

    # Plot trajectories from these specific initial states
    plot_trajectories_with_values(model, initial_states=initial_states, num_steps=200)


if __name__ == "__main__":
    main()
