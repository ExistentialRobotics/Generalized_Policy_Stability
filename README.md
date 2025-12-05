# Certifying Stability of Reinforcement Learning Policies using Generalized Lyapunov Functions

## Environment setup

We recommend creating the Conda environment defined in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate generalized_neural_stability
```

## Linear systems (LQR stability)

```bash
# Multi-step certification 
python linear_system/main_linear.py scalar

# Find optimal discount factor γ
python linear_system/find_optimal_gamma.py scalar
```

Result figures will be saved in `results/linear_system/`:
- `gamma_bounds_scalar.png` - Certified γ bounds vs number of steps M
- `line_gamma_M2_sum2_scalar.png` - Parameter sweep for M=2 case

## Nonlinear systems (RL stability)

### Gym Inverted Pendulum

Train and evaluate RL policies (PPO/SAC) on the Gymnasium inverted pendulum:

```bash
# Train a policy
python nonlinear_system/gym_inverted_pendulum/main_rl.py --algo ppo --timesteps 500000

# Evaluate a trained policy
python nonlinear_system/gym_inverted_pendulum/main_rl.py --algo ppo --evaluate
```

Train generalized Lyapunov functions for stability certification:

```bash
# Train with step weights (multi-step certification)
python nonlinear_system/gym_inverted_pendulum/training/step_weights_training.py

# Visualize Lyapunov functions and trajectories
python nonlinear_system/gym_inverted_pendulum/training/step_weights_visual.py
```

Result figures are saved in `results/gym_inverted_pendulum/{algo}/`:
- `total_lyapunov_with_trajectories.png` - Lyapunov function contour with trajectories
- `value_differences.png` - Stability condition violations across state space
- `lyapunov_values_over_time.png` - Lyapunov function evolution along trajectories

## Citation

If you found this work useful, we would appreciate if you could cite our work:

BibTeX (arXiv):
```bibtex
@article{long2025certifying,
  title={Certifying stability of reinforcement learning policies using generalized lyapunov functions},
  author={Long, Kehan and Cort{\'e}s, Jorge and Atanasov, Nikolay},
  journal={arXiv preprint arXiv:2505.10947},
  year={2025}
}
```
