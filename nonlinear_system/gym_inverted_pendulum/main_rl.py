#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for training RL algorithms on pendulum environment.
"""

import argparse
import os
import torch
import traceback
import gymnasium as gym
from stable_baselines3 import PPO, SAC

# Import functions from rl_baseline
from training.rl_baseline import evaluate_rl_controller, test_model, plot_actions


def ensure_directory(directory):
    """Ensure a directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RL Training for Pendulum Control")
    parser.add_argument(
        "--algo", type=str, default="ppo", choices=["ppo", "sac"], help="RL algorithm to use"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="nonlinear_system/gym_inverted_pendulum/saved_models",
        help="Directory to save models",
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate and visualize after training"
    )
    parser.add_argument(
        "--test_steps", type=int, default=300, help="Number of steps to test the model"
    )
    parser.add_argument(
        "--use_modified_env",
        action="store_true",
        help="Use modified Gymnasium environment instead of standard one",
    )
    parser.add_argument(
        "--timesteps", type=int, default=500000, help="Number of timesteps to train"
    )
    return parser.parse_args()


def main():
    """Main function for RL training."""
    args = parse_args()

    # Ensure directories exist
    ensure_directory(args.save_dir)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up environment
    if args.use_modified_env:
        from Gymnasium_modified.gymnasium.envs.classic_control.pendulum import PendulumEnv

        test_env = PendulumEnv(render_mode="human", g=9.81, m=1.0, l=1.0, b=0.13)
        print("Using modified Gymnasium environment")
    else:
        test_env = gym.make("Pendulum-v1", render_mode="human")
        print("Using standard Gymnasium environment")

    # Determine model paths
    standard_model_path = os.path.join(args.save_dir, args.algo, "pendulum.zip")
    model_path = os.path.join(args.save_dir, f"{args.algo}_pendulum")
    env_suffix = "_modified" if args.use_modified_env else "_standard"
    model_path += env_suffix

    model = None
    load_path = None

    # Check if model exists in either location
    if os.path.exists(standard_model_path):
        load_path = standard_model_path.replace(".zip", "")
    elif os.path.exists(f"{model_path}.zip"):
        load_path = model_path

    # Load existing model if available, otherwise train
    if load_path is not None:
        # Load existing model
        print(f"Loading existing model from: {load_path}.zip")
        if args.algo == "ppo":
            model = PPO.load(load_path)
        else:  # SAC
            model = SAC.load(load_path)
        
        if not args.evaluate:
            print("Model loaded. Use --evaluate flag to evaluate it.")
    else:
        # No existing model: train a new one
        if args.evaluate:
            print("Error: No trained model found for evaluation.")
            print(f"  Checked: {standard_model_path}")
            print(f"  Checked: {model_path}.zip")
            print("Please train a model first (run without --evaluate flag).")
            return
        
        print(f"Training {args.algo} on pendulum...")
        if args.use_modified_env:
            train_env = PendulumEnv(g=9.81, m=1.0, l=1.0, b=0.13)
        else:
            train_env = gym.make("Pendulum-v1")

        # Create and train model
        if args.algo == "ppo":
            model = PPO("MlpPolicy", train_env, verbose=1)
        else:  # SAC
            model = SAC("MlpPolicy", train_env, verbose=1)

        model.learn(total_timesteps=args.timesteps)
        model.save(model_path)
        print(f"Model saved to {model_path}.zip")

    if args.evaluate and model is not None:
        try:

            print("Evaluating model:", model_path)

            # Generate value function and policy plots
            try:
                evaluate_rl_controller(model)
            except Exception as e:
                print(f"Warning: Could not run visual evaluation: {e}")
                traceback.print_exc()

            # Test the model
            print("Testing model...")
            try:
                actions, values = test_model(model, test_env, num_steps=args.test_steps)
                plot_actions(
                    values,
                    values,
                    f"{args.algo.upper()} Value Function Over Time ({env_suffix})",
                    os.path.join(args.save_dir, f"value_function_{args.algo}{env_suffix}.png"),
                    y_axis_name="Value",
                )
                plot_actions(
                    actions,
                    actions,
                    f"{args.algo.upper()} Actions Over Time ({env_suffix})",
                    os.path.join(args.save_dir, f"actions_{args.algo}{env_suffix}.png"),
                    y_axis_name="Control Input",
                )
            except Exception as e:
                print(f"Warning: Could not run testing: {e}")
                traceback.print_exc()

        except Exception as e:
            print(f"Error loading or evaluating model: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
