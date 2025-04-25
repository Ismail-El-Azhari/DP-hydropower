import numpy as np
from runoff.transition_matrices import transition_matrices
from runoff.runoff_data import runoff_qr
from SDP.utils import *
import os

def sdp_dam(max_iter=100, tol=0.1, Print_Iterations=True, debug=False):
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

            runoff_values = runoff_qr[month]   # list of 10 QR bins
            M = len(runoff_values)

            # on first iteration only evaluate one state to speed up startup
            if iteration == 0 and month == "June":
                zb_indices = [765 - Z_MIN]
            else:
                zb_indices = range(len(ZB_LEVELS))

            for zb_i in zb_indices:
                zb = ZB_LEVELS[zb_i]
                avg_value = 0.0

                for qr_idx1 in range(M):
                    best_value = -np.inf
                    best_release = 0

                    for release in RELEASES:
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
                J[month][zb_i] = avg_value / M

        # check convergence
        delta = max(np.max(np.abs(J[m] - J_prev[m])) for m in months)
        if Print_Iterations:
            print(f" Iteration {iteration+1}, delta = {delta:.6f}")
        if delta < tol:
            print(f" Converged after {iteration+1} iterations.")
            break

    # Save both main policy and the detailed per-QR policy arrays
    os.makedirs("policy2_npy", exist_ok=True)

    for m, P in policy_by_qr.items():
        np.save(f"policy2_npy/policy_{m}_qr.npy", P)

    return policy, J
