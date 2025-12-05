#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare original and refined pendulum policies.
"""

import os
import sys
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import torch
from network.PolicyNet import PolicyNet

# Add the parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_policy(path, device):
    """Load a PolicyNet model from path."""
    # Create policy with matching architecture
    policy = PolicyNet(n_input=3, n_hidden=16, n_output=1, n_layers=5).to(device)

    try:
        state_dict = torch.load(path, map_location=device)
        policy.load_state_dict(state_dict)
        print(f"Loaded policy from {path}")
        return policy
    except Exception as e:
        print(f"Error loading policy from {path}: {e}")
        return None


def get_action(policy, state, device):
    """Get action from policy for a given state."""
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).to(device)
        action = policy(state_tensor).cpu().numpy()
        return action  # Clip to pendulum action range


def plot_policies_comparison(original_policy, refined_policy, save_dir):
    """Plot policies in the same style as evaluate_rl_controller."""
    # Create a grid of states
    theta_bins = 100
    omega_bins = 100
    theta = np.linspace(-np.pi * 2, np.pi * 2, theta_bins)
    omega = np.linspace(-8, 8, omega_bins)
    Theta, Omega = np.meshgrid(theta, omega)

    # Prepare flattened states
    cos_theta = np.cos(Theta.ravel())
    sin_theta = np.sin(Theta.ravel())
    omega_flat = Omega.ravel()

    # Initialize arrays for actions
    original_actions = np.zeros_like(Theta.ravel())
    refined_actions = np.zeros_like(Theta.ravel())

    # Get device from policies
    device = next(original_policy.parameters()).device

    # Evaluate both policies
    for i in range(len(Theta.ravel())):
        state = np.array([cos_theta[i], sin_theta[i], omega_flat[i]])
        state_tensor = torch.FloatTensor(state).to(device)

        # Get actions from both policies
        with torch.no_grad():
            # Original policy
            original_action = original_policy(state_tensor.unsqueeze(0))
            original_action = torch.clamp(original_action, -2.0, 2.0)

            # Refined policy
            refined_action = refined_policy(state_tensor.unsqueeze(0))
            refined_action = torch.clamp(refined_action, -2.0, 2.0)

        original_actions[i] = original_action.cpu().numpy()
        refined_actions[i] = refined_action.cpu().numpy()

    # Reshape arrays
    original_actions = original_actions.reshape(Theta.shape)
    refined_actions = refined_actions.reshape(Theta.shape)

    # Create comparison plot
    plt.figure(figsize=(14, 6))

    # Original policy
    plt.subplot(1, 2, 1)
    contour = plt.contourf(Theta, Omega, original_actions, levels=50, cmap="viridis")
    plt.colorbar(contour)
    plt.xlabel("Theta (radians)", fontsize=16)
    plt.ylabel("Omega (rad/s)", fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title("Original Policy", fontsize=18)

    # Refined policy
    plt.subplot(1, 2, 2)
    contour = plt.contourf(Theta, Omega, refined_actions, levels=50, cmap="viridis")
    plt.colorbar(contour)
    plt.xlabel("Theta (radians)", fontsize=16)
    plt.ylabel("Omega (rad/s)", fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title("Refined Policy", fontsize=18)

    plt.suptitle("Policy Comparison", fontsize=18, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "policy_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set paths to models
    original_model_path = "inverted_pendulum/saved_models/mimic_ppo_policy.pth"
    refined_model_path = "inverted_pendulum/saved_models/20_steps_controller.pt"

    # Create output directory for plots
    output_dir = "inverted_pendulum/results/policy_comparison"
    os.makedirs(output_dir, exist_ok=True)

    # Load models
    original_policy = load_policy(original_model_path, device)
    refined_policy = load_policy(refined_model_path, device)

    if original_policy is None or refined_policy is None:
        print("Failed to load policies. Exiting.")
        return

    # Set both policies to evaluation mode
    original_policy.eval()
    refined_policy.eval()

    # Generate comparison plots
    print("Generating policy comparison plots...")
    plot_policies_comparison(original_policy, refined_policy, output_dir)

    print(f"\nComparison complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
