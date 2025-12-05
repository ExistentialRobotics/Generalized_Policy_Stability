import os

import cvxpy as cp
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_system_from_yaml(yaml_path, system_name):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    if system_name not in data:
        raise ValueError(f"System '{system_name}' not found in {yaml_path}")
    sys_data = data[system_name]
    # Convert lists to np.array
    A = np.array(sys_data["A"])
    B = np.array(sys_data["B"])
    Q = np.array(sys_data["Q"])
    R = np.array(sys_data["R"])
    weights = sys_data.get("weights", None)
    true_gamma = sys_data.get("true_gamma", None)
    return A, B, Q, R, weights, true_gamma


def solve_riccati_iterative(A, B, Q, R, max_iter=5000, tolerance=1e-10):
    """
    Solve the continuous-time algebraic Riccati equation iteratively:
    P = Q + A^T(P - PB(R + B^TPB)^(-1)B^TP)A
    """
    P = Q  # Initial guess

    for i in range(max_iter):
        # Store previous P for convergence check
        P_prev = P.copy()

        # Compute the next iteration
        BTP = B.T @ P
        BTPB = BTP @ B
        R_BTPB = R + BTPB
        R_BTPB_inv = np.linalg.inv(R_BTPB)
        middle_term = P @ B @ R_BTPB_inv @ BTP
        P_minus_middle = P - middle_term
        final_term = A.T @ P_minus_middle @ A
        P = Q + final_term

        # Check convergence
        error = np.max(np.abs(P - P_prev))
        if error < tolerance:
            print(f"Converged after {i + 1} iterations")
            return P

    print("Warning: Maximum iterations reached without convergence")

    RHS_, error = verify_solution(A, B, Q, R, P)
    if error > 1e-5:
        print(f"Error: {error}")
        print(f"RHS: {RHS_}")
        print(f"P: {P}")
        raise ValueError("Solution does not satisfy the Riccati equation")
    return P


def verify_solution(A, B, Q, R, P):
    """Verify if P satisfies the Riccati equation"""
    BTP = B.T @ P
    BTPB = BTP @ B
    R_BTPB = R + BTPB
    R_BTPB_inv = np.linalg.inv(R_BTPB)
    middle_term = P @ B @ R_BTPB_inv @ BTP
    P_minus_middle = P - middle_term
    final_term = A.T @ P_minus_middle @ A
    RHS = Q + final_term

    error = np.max(np.abs(P - RHS))
    return RHS, error


def create_lmi_constraint(A, B, Q, R, S_matrices, varpi, coefficient, constraint_type):
    """
    Create an LMI constraint based on the type and coefficient

    Args:
        constraint_type: "type1" or "type2"
        coefficient: The coefficient (sigma_i/M)
        S_matrices: List of S matrices [S1, Si+1] for type1 or [S1, S2] for type2
    """
    if constraint_type == "type1":
        # For k+i steps: (sigma_{i+1}/M)A'S₁A - Sᵢ₊₂ - varpi * Q
        top_left = coefficient * A.T @ S_matrices[0] @ A - S_matrices[1] - varpi * Q
        top_right = coefficient * A.T @ S_matrices[0] @ B
        bottom_right = coefficient * B.T @ S_matrices[0] @ B - varpi * R
    else:  # type2
        # For k step: (sigma_1/M)A'S₁A - S₁ + S₂ - varpi * Q
        top_left = coefficient * A.T @ S_matrices[0] @ A - S_matrices[0] + S_matrices[1] - varpi * Q
        top_right = coefficient * A.T @ S_matrices[0] @ B
        bottom_right = coefficient * B.T @ S_matrices[0] @ B - varpi * R

    return cp.bmat([[top_left, top_right], [top_right.T, bottom_right]])


def build_lmi_block(A, B, S0, S1, Q, R, varpi, sigma, M, lmi_type):
    """
    Build the block LMI for the generalized Lyapunov condition.
    lmi_type: 'initial' or 'future'
    """
    if lmi_type == "initial":
        # For the initial step (k)
        coef = sigma / M
        top_left = coef * A.T @ S0 @ A - S0 + S1 - varpi * Q
        top_right = coef * A.T @ S0 @ B
        bottom_left = coef * B.T @ S0 @ A
        bottom_right = coef * B.T @ S0 @ B - varpi * R
    elif lmi_type == "future":
        # For future steps (k+i)
        coef = sigma / M
        top_left = coef * A.T @ S0 @ A - S1 - varpi * Q
        top_right = coef * A.T @ S0 @ B
        bottom_left = coef * B.T @ S0 @ A
        bottom_right = coef * B.T @ S0 @ B - varpi * R
    else:
        raise ValueError("lmi_type must be 'initial' or 'future'")
    return cp.bmat([[top_left, top_right], [bottom_left, bottom_right]])


def solve_lmi_n_step(A, B, Q, R, P, n_steps, sigmas=None):
    """
    Solve the M-step LMI optimization problem

    Args:
        n_steps (int): Number of steps (>= 1)
        sigmas (list): List of sigma values. If None, will use default values
    """
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")

    # Generate default sigmas if not provided
    if sigmas is None:
        sigmas = [1.0] * n_steps  # Default to 1.0 for each step

    if len(sigmas) != n_steps:
        raise ValueError("Number of sigma values must match n_steps")

    # print(f"Solving {n_steps}-step LMI with sigmas: {sigmas}")

    # Get dimensions
    n = A.shape[0]

    # Step 1: Maximize minimum alpha
    # print("Step 1: Maximizing min(alphas)...")

    # Create variables
    S_vars = [cp.Variable((n, n), symmetric=True) for _ in range(n_steps + 1)]
    varpi = cp.Variable()
    alphas = [cp.Variable() for _ in range(n_steps)]
    alpha_min = cp.Variable()

    # Basic constraints
    constraints = []
    for S in S_vars:
        constraints.append(S >> 0)
    constraints.extend([varpi >= 0, alpha_min >= 0])

    # Alpha constraints
    for alpha in alphas:
        constraints.extend([alpha >= 0, alpha >= alpha_min])

    # Create LMI constraints
    if n_steps == 1:
        # For one step, we only need the type2 LMI
        coef = sigmas[0] / n_steps
        lmi = create_lmi_constraint(A, B, Q, R, [S_vars[0], S_vars[1]], varpi, coef, "type2")
        constraints.append(lmi <= 0)
    else:
        # First add the k step (type2) LMI
        coef = sigmas[0] / n_steps
        lmi = create_lmi_constraint(A, B, Q, R, [S_vars[0], S_vars[1]], varpi, coef, "type2")
        constraints.append(lmi <= 0)

        # Then add k+i steps (type1) LMIs
        for i in range(n_steps - 1):
            coef = sigmas[i + 1] / n_steps
            lmi = create_lmi_constraint(
                A, B, Q, R, [S_vars[0], S_vars[i + 2]], varpi, coef, "type1"
            )
            constraints.append(lmi <= 0)

    # S matrix conditions
    for i in range(n_steps):
        constraints.append(alphas[i] * P <= S_vars[i + 1])

    # Numerical stability bounds
    constraints.extend([alpha_min <= 1, varpi <= 1e3])

    # Solve step 1
    prob1 = cp.Problem(cp.Maximize(alpha_min), constraints)
    prob1.solve()

    optimal_alphas = [alpha.value for alpha in alphas]
    _ = alpha_min.value

    # print(f"Step 1 complete. Optimal alpha: {optimal_alphas}")
    # print(f"Step 1 complete. Optimal alpha_min: {optimal_alpha_min}")

    # Step 2: Minimize varpi while maintaining alphas
    # print("\nStep 2: Minimizing varpi...")

    # Create new variables for step 2
    S_vars_2 = [cp.Variable((n, n), symmetric=True) for _ in range(n_steps + 1)]
    varpi_2 = cp.Variable()

    # Basic constraints for step 2
    constraints_2 = []
    for S in S_vars_2:
        constraints_2.append(S >> 0)
    constraints_2.append(varpi_2 >= 0)

    # Maintain the optimal alphas from step 1
    for i in range(n_steps):
        constraints_2.append(optimal_alphas[i] * P <= S_vars_2[i + 1])

    # Create LMI constraints for step 2
    if n_steps == 1:
        coef = sigmas[0]
        lmi = create_lmi_constraint(A, B, Q, R, [S_vars_2[0], S_vars_2[1]], varpi_2, coef, "type2")
        constraints_2.append(lmi <= 0)
    else:
        for i in range(n_steps - 1):
            coef = sigmas[i + 1] / n_steps
            lmi = create_lmi_constraint(
                A, B, Q, R, [S_vars_2[0], S_vars_2[i + 2]], varpi_2, coef, "type1"
            )
            constraints_2.append(lmi <= 0)

        coef = sigmas[0] / n_steps
        lmi = create_lmi_constraint(A, B, Q, R, [S_vars_2[0], S_vars_2[1]], varpi_2, coef, "type2")
        constraints_2.append(lmi <= 0)

    # Numerical stability bound
    constraints_2.append(varpi_2 <= 1e3)

    # Solve step 2
    prob2 = cp.Problem(cp.Minimize(varpi_2), constraints_2)
    prob2.solve()

    if prob2.status != "optimal":
        raise ValueError(f"Step 2 failed. Status: {prob2.status}")

    return {
        "alphas": optimal_alphas,
        "varpi": varpi_2.value,
        "sigmas": sigmas,
        "S_matrices": [S.value for S in S_vars_2],
        "gamma": compute_gamma(optimal_alphas, varpi_2.value, sigmas),
        "objective_value": prob1.value,
    }


def compute_gamma(alphas, varpi, sigmas):
    """
    Compute the final gamma value based on the M-step solution
    """
    M = len(alphas)
    gamma_terms = []

    # Process terms from i=2 to M
    for i in range(1, M):
        gamma_terms.append(sigmas[i] * varpi / (M * alphas[i]))

    # Add the k-step term (i=1)
    gamma_terms.append(sigmas[0] * varpi / (M * (varpi + alphas[0])))

    return max(gamma_terms)


def analyze_multiple_steps(A, B, Q, R, weights=None):
    """
    Analyze the system for M=1..10, using provided weights if available.
    """
    P_iterative = solve_riccati_iterative(A, B, Q, R)
    gamma_values = []
    for M in range(1, 11):
        if weights and len(weights) >= M:
            sigmas = weights[M - 1]
        else:
            sigmas = [1.0] * M
        print(f"\nAnalyzing M={M} case...")
        solution = solve_lmi_n_step(A, B, Q, R, P_iterative, n_steps=M, sigmas=sigmas)
        gamma_values.append(solution["gamma"])
        print(f"M={M}: γ={solution['gamma']:.6f}")
    return gamma_values


def main():
    system_name = "scalar"  # default

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, "systems.yaml")

    A, B, Q, R, weights, true_gamma = load_system_from_yaml(yaml_path, system_name)
    print(f"Running analysis for system: {system_name}")

    gamma_values = analyze_multiple_steps(A, B, Q, R, weights)
    M_values = range(1, 11)
    print("\nAll gamma values:", gamma_values)

    # Ensure results directory exists (inside project root)
    results_dir = os.path.join(os.path.dirname(script_dir), "results", "linear_system")
    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Saving figures to: {results_dir}")

    plt.figure(figsize=(8, 5))
    plt.plot(M_values, gamma_values, "bo-", label="Certified γ bounds", linewidth=3, markersize=8)
    if true_gamma is not None:
        plt.axhline(y=true_gamma, color="r", linestyle="--", label="True γ bound", linewidth=3)
    plt.xlabel("Number of steps (M)", fontsize=24)
    plt.ylabel("γ bound", fontsize=24)
    # plt.title(f'Linear System ({system_name}): Computed γ Bounds')
    plt.grid(True)
    plt.legend(fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0.3, max(gamma_values) * 1.1)
    plt.tight_layout()
    bounds_path = os.path.join(results_dir, f"gamma_bounds_{system_name}.png")
    plt.savefig(bounds_path)

    # Second figure: gamma vs sigma1 with sigma1+sigma2=2 for M=2
    P_iterative = solve_riccati_iterative(A, B, Q, R)
    sweep_path = os.path.join(results_dir, f"line_gamma_M2_sum2_{system_name}.png")
    from utils_visual import line_gamma_M2_sum2

    line_gamma_M2_sum2(A, B, Q, R, P_iterative, num_points=200, save_path=sweep_path)


if __name__ == "__main__":
    main()
