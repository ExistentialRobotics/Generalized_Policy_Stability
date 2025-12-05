import matplotlib.pyplot as plt
import numpy as np

# Import your Riccati/LMI solvers and system definitions
from main_linear import solve_lmi_n_step, solve_riccati_iterative


def plot_gamma_vs_M(M_list, gamma_list, optimal_gamma=None, save_path=None):
    """
    Plot gamma lower bound vs number of steps M.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(M_list, gamma_list, "bo-", label="Certified γ bound")
    if optimal_gamma is not None:
        plt.axhline(y=optimal_gamma, color="r", linestyle="--", label="True threshold")
    plt.xlabel("Number of Steps $M$", fontsize=24)
    plt.ylabel(r"Certified $\gamma$ Bound", fontsize=24)
    # plt.title('Certified $\gamma$ Bound vs. Number of Steps $M$')
    plt.grid(True)
    plt.legend(fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")


def simulate_closed_loop(A, B, K, x0, steps):
    """
    Simulate closed-loop system: x_{k+1} = (A - B K) x_k
    """
    n = x0.shape[0]
    xs = np.zeros((steps + 1, n))
    xs[0] = x0
    for k in range(steps):
        xs[k + 1] = (A - B @ K) @ xs[k]
    return xs


def plot_lyapunov_trajectories(
    A, B, Q, R, M_list, sigmas_list, gamma_list, steps=20, save_path=None
):
    """
    Plot the decrease of the composite Lyapunov function Y_gamma(x_k) for different M.
    """
    np.random.seed(42)
    x0 = np.random.randn(A.shape[0])
    plt.figure(figsize=(8, 5))
    for M, sigmas, gamma in zip(M_list, sigmas_list, gamma_list):
        P = solve_riccati_iterative(A, B, Q, R)
        # Compute optimal K for discounted Riccati (approximate)
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        xs = simulate_closed_loop(A, B, K, x0, steps)
        # Compute composite Lyapunov Y_gamma(x) = x^T P x + x^T S0 x / varpi
        # For illustration, use S0 = P, varpi = 1
        S0 = P
        varpi = 1.0
        Y_vals = [x.T @ P @ x + (x.T @ S0 @ x) / varpi for x in xs]
        plt.plot(range(steps + 1), Y_vals, label=f"M={M}")
    plt.xlabel("Time step $k$")
    plt.ylabel(r"$Y_\gamma(x_k)$")
    plt.title("Composite Lyapunov Function Trajectories")
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")


def line_gamma_M2_sum2(A, B, Q, R, P, num_points=200, save_path=None):
    """
    Plot gamma bound as a function of sigma_1, with sigma_1 + sigma_2 = 2.
    Highlights the minimum gamma and the classical LF case (sigma_1=2, sigma_2=0).
    Gamma values are capped at 1.0.
    """
    sigma1_vals = np.linspace(0, 2, num_points)
    gamma_vals = []
    feasible_count = 0
    for sigma1 in sigma1_vals:
        sigma2 = 2 - sigma1
        sigma1 = float(sigma1)
        sigma2 = float(sigma2)
        try:
            sol = solve_lmi_n_step(A, B, Q, R, P, n_steps=2, sigmas=[sigma1, sigma2])
            gamma = sol["gamma"]
            if gamma is not None and np.isfinite(gamma):
                # Cap gamma at 1.0
                gamma = min(gamma, 1.0)
                gamma_vals.append(gamma)
                feasible_count += 1
            else:
                gamma_vals.append(np.nan)
        except Exception:
            gamma_vals.append(np.nan)
    print(f"Feasible points: {feasible_count} / {num_points}")

    gamma_vals = np.array(gamma_vals)
    plt.figure(figsize=(8, 5))  # Match the size of the other figure
    plt.plot(sigma1_vals, gamma_vals, "b-", linewidth=3, label=r"$\sigma_1 + \sigma_2 = 2$")

    # Highlight minimum gamma
    if np.any(np.isfinite(gamma_vals)):
        min_idx = np.nanargmin(gamma_vals)
        min_sigma1 = sigma1_vals[min_idx]
        min_gamma = gamma_vals[min_idx]
        plt.plot(min_sigma1, min_gamma, "r*", markersize=16, label=r"Min Certified $\gamma$")
        # Annotation for min point (optional)

    # Highlight classical LF case (sigma1=2, sigma2=0)
    idx_classical = np.abs(sigma1_vals - 2).argmin()
    gamma_classical = gamma_vals[idx_classical]
    plt.plot(2, gamma_classical, "go", markersize=14, label=r"Classical LF ($\sigma_1=2$)")

    plt.xlabel(r"$\sigma_1$", fontsize=24)
    plt.ylabel(r"Certified $\gamma$ Bound", fontsize=24)
    plt.grid(True)
    plt.legend(fontsize=20)
    plt.xticks([0.0, 0.5, 1.0, 1.5, 2.0], fontsize=22)
    plt.yticks(np.arange(0.6, 1.05, 0.1), fontsize=22)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
