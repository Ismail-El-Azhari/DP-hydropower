import numpy as np
from runoff.transition_matrices import transition_matrices
from runoff.runoff_data import runoff_qr
from SDP.utils import *
import os
import random  # For random sampling

def adp_dam(max_iter=100, tol=0.01, Print_Iterations=True, debug=False,
                   n_zb_samples=90,  # Number of ZB states to sample
                   n_qr_samples=5,    # Number of QR samples for expectation
                   n_release_samples=5): # Number of release actions to sample
    """
    Approximate Stochastic Dynamic Programming for dam management.  This version
    uses state and action sampling to reduce the computational burden associated
    with the curse of dimensionality.  It calculates the final policy and value 
    function by averaging over QR bins.

    Args:
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.(I chose 0.1 for faster convergence and it makes sense regarding the values of J)
        Print_Iterations (bool): To print delta to see how fast we converge
        debug (bool):  To help with debugging, can be removed
        n_zb_samples (int): Number of ZB (reservoir level) states to sample in each month.
        n_qr_samples (int): Number of QR (runoff quantile) samples to use when
            approximating the expectation over future states.
        n_release_samples (int): Number of release actions to sample.

    Returns:
        tuple: (policy, J)
            policy (dict):  Approximate optimal policy for each month.
            J (dict): Approximate optimal value function for each month.
    """
    # Initialize value and policy tables
    J = {month: np.zeros(len(ZB_LEVELS)) for month in months}
    policy = {month: np.zeros(len(ZB_LEVELS)) for month in months}
    policy_by_qr = {month: np.zeros((10, len(ZB_LEVELS))) for month in months}  # full QR policy for visualization
    beta = 0.9  # Discount factor

    for iteration in range(max_iter):
        # snapshot of old values for convergence check
        J_prev = {m: J[m].copy() for m in months}

        # backward DP sweep
        for i in reversed(range(len(months))):
            month = months[i]
            next_month = months[(i + 1) % 12]

            # determine transition matrix key
            curr_num = (i + 6) % 12 or 12
            next_num = (curr_num % 12) + 1
            P = transition_matrices[f"{curr_num}_{next_num}"]

            runoff_values = runoff_qr[month]
            M = len(runoff_values)

            # Sample ZB states.
            zb_indices = random.sample(range(len(ZB_LEVELS)), min(n_zb_samples, len(ZB_LEVELS)))

            for zb_i in zb_indices:
                zb = ZB_LEVELS[zb_i]
                avg_value = 0.0

                #for qr_idx1 in range(M): # changed
                qr_indices = np.linspace(0, M-1, n_qr_samples, dtype=int)
                for qr_idx1 in qr_indices:
                    best_value = -np.inf
                    best_release = 0

                    # Sample releases
                    sampled_releases = np.linspace(min(RELEASES), max(RELEASES), n_release_samples)
                    for release in sampled_releases:
                        ze = zb - release
                        if ze < Z_MIN:
                            continue

                        reward = power(runoff_values[qr_idx1], zb, ze)
                        penalty = -1000 if (ze < 750 or ze > 830) else 0

                        expected_value = 0.0
                        for qr_idx2 in range(M):
                            prob = P[qr_idx1][qr_idx2]
                            runoff_next = runoff_qr[next_month][qr_idx2]
                            next_zb = ze + runoff_next

                            if not (Z_MIN <= next_zb <= Z_MAX):
                                continue

                            next_zb_i = int(round(next_zb)) - Z_MIN
                            future = J_prev[next_month][next_zb_i]
                            expected_value += prob * (reward + penalty + beta * future)

                        if expected_value > best_value:
                            best_value = expected_value
                            best_release = release

                    policy_by_qr[month][qr_idx1, zb_i] = best_release
                    avg_value += best_value

                # After looping all QR bins, store average value and policy
                policy[month][zb_i] = np.mean(policy_by_qr[month][:, zb_i])
                J[month][zb_i] = avg_value / len(qr_indices) # changed

        # check convergence
        delta = max(np.max(np.abs(J[m] - J_prev[m])) for m in months)
        if Print_Iterations:
            print(f" Iteration {iteration+1}, delta = {delta:.6f}")
        if delta < tol:
            print(f" Converged after {iteration+1} iterations.")
            break

    # Save both main policy and the detailed per-QR policy arrays
    os.makedirs("policy_adp_npy", exist_ok=True)

    for m, P in policy_by_qr.items():
        np.save(f"policy_adp_npy/policy_{m}_approx.npy", P) # changed

    return policy, J
