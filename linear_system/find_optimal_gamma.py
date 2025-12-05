import os

import numpy as np
import yaml


def load_system_from_yaml(yaml_path, system_name):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    if system_name not in data:
        raise ValueError(f"System '{system_name}' not found in {yaml_path}")
    sys_data = data[system_name]
    A = np.array(sys_data["A"])
    B = np.array(sys_data["B"])
    Q = np.array(sys_data["Q"])
    R = np.array(sys_data["R"])
    true_gamma = sys_data.get("true_gamma", None)
    return A, B, Q, R, true_gamma


def find_optimal_gamma_discounted(A, B, Q, R, tol=1e-8, verbose=False):
    """
    Find the minimal gamma in (0,1) such that the closed-loop system is stable
    for the discounted LQR problem: J = sum gamma^k (x_k^T Q x_k + u_k^T R u_k)
    Returns: optimal_gamma, K_gamma, F_gamma
    """

    def is_stable(gamma):
        try:
            P = solve_discounted_riccati(A, B, Q, R, gamma)
            if not np.all(np.isfinite(P)):
                return False, None, None
            K = discounted_lqr_gain(A, B, Q, R, P, gamma)
            F = A + B @ K
            eigs = np.linalg.eigvals(F)
            return np.all(np.abs(eigs) < 1), K, F
        except Exception as e:
            if verbose:
                print(f"gamma={gamma:.10f}, Riccati failed: {e}")
            return False, None, None

    left, right = 0.0, 1.0
    best_gamma = None
    best_K = None
    best_F = None
    while right - left > tol:
        mid = (left + right) / 2
        stable, K, F = is_stable(mid)
        if verbose:
            eigs_str = np.array2string(np.linalg.eigvals(F)) if F is not None else "None"
            print(f"gamma={mid:.10f}, stable={stable}, eigs={eigs_str}")
        if stable:
            right = mid
            best_gamma, best_K, best_F = mid, K, F
        else:
            left = mid
    return best_gamma, best_K, best_F


def solve_discounted_riccati(A, B, Q, R, gamma, max_iter=5000, tolerance=1e-10):
    """
    Solve the discounted continuous-time algebraic Riccati equation:
    A^T P_γ - γ^2 P_γ B(R + B^T P_γ B)^(-1) B^T P_γ A + Q = 0

    Args:
        gamma (float): Discount factor between 0 and 1
    """
    P = Q  # Initial guess

    for i in range(max_iter):
        P_prev = P.copy()

        # Compute the next iteration with discount factor
        BTP = B.T @ P
        BTPB = gamma * BTP @ B
        R_BTPB = R + BTPB
        R_BTPB_inv = np.linalg.inv(R_BTPB)
        middle_term = gamma * gamma * P @ B @ R_BTPB_inv @ BTP
        final_term = A.T @ (gamma * P - middle_term) @ A
        P = Q + final_term

        # Check convergence
        error = np.max(np.abs(P - P_prev))
        if error < tolerance:
            print(f"Discounted Riccati converged after {i + 1} iterations")
            return P

    print("Warning: Maximum iterations reached without convergence")
    return P


def discounted_lqr_gain(A, B, Q, R, P, gamma):
    """
    Compute the optimal discounted LQR gain:
    K = - (gamma * B^T P B + R)^{-1} B^T P A
    """
    return -np.linalg.inv(gamma * B.T @ P @ B + R) @ (B.T @ P @ A)


if __name__ == "__main__":
    # Usage: python find_optimal_gamma.py [system_name]

    system_name = "scalar"  # default
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, "systems.yaml")

    A, B, Q, R, true_gamma = load_system_from_yaml(yaml_path, system_name)
    print(f"Finding optimal gamma for system: {system_name}, A: {A}, B: {B}, Q: {Q}, R: {R}")

    opt_gamma, opt_K, opt_F = find_optimal_gamma_discounted(A, B, Q, R, tol=1e-8, verbose=True)
    print(f"\n[Discounted Riccati] {system_name} system optimal gamma: {opt_gamma:.10f}")

    print(f"Optimal feedback K: {opt_K}")
    print(f"Closed-loop eigenvalues: {np.linalg.eigvals(opt_F)}")
