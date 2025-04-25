import numpy as np
from runoff.transition_matrices import transition_matrices
from runoff.runoff_data import runoff_qr
from SDP.utils import *
import os

from numpy.polynomial.polynomial import Polynomial


def adp_dam(max_iter=100, tol=0.1, degree=2, Print_Iterations=True, alpha=0.4):
    beta = 0.9  # discount factor

    # Initialize value function J as coefficients of polynomial basis: J(zb) ≈ w0 + w1*zb + w2*zb²
    coeffs = {month: np.zeros(degree + 1) for month in months}
    policy = {month: np.zeros(len(ZB_LEVELS)) for month in months}

    for iteration in range(max_iter):
        coeffs_prev = {m: coeffs[m].copy() for m in months}

        # Value samples used to refit polynomial after iteration
        sampled_J = {month: np.zeros(len(ZB_LEVELS)) for month in months}

        for i in reversed(range(len(months))):
            month = months[i]
            next_month = months[(i + 1) % 12]

            curr_num = (i + 6) % 12 or 12
            next_num = (curr_num % 12) + 1
            P = transition_matrices[f"{curr_num}_{next_num}"]

            runoff_values = runoff_qr[month]
            M = len(runoff_values)

            for zb_i, zb in enumerate(ZB_LEVELS):
                best_value = -np.inf
                best_release = 0

                for release in RELEASES:
                    ze = zb - release
                    if ze < Z_MIN:
                        continue

                    expected_value = 0.0
                    for qr_idx1 in range(M):
                        reward = power(runoff_values[qr_idx1], zb, ze)
                        penalty = -1000 if (ze < 750 or ze > 830) else 0

                        row_sum = 0.0
                        for qr_idx2 in range(M):
                            prob = P[qr_idx1][qr_idx2]
                            runoff_next = runoff_qr[next_month][qr_idx2]
                            next_zb = ze + runoff_next
                            if not (Z_MIN <= next_zb <= Z_MAX):
                                continue

                            # Use polynomial value approximation
                            J_next = Polynomial(coeffs_prev[next_month])(next_zb)
                            row_sum += prob * (reward + penalty + beta * J_next)

                        expected_value += row_sum / M

                    if expected_value > best_value:
                        best_value = expected_value
                        best_release = release

                sampled_J[month][zb_i] = best_value
                policy[month][zb_i] = best_release

        # Fit a new polynomial to the sampled values with smoothing
        for month in months:
            fitted_coeffs = Polynomial.fit(ZB_LEVELS, sampled_J[month], deg=degree).convert().coef
            coeffs[month] = (1 - alpha) * coeffs_prev[month] + alpha * fitted_coeffs

        # Convergence check using coefficient distance
        delta = max(np.linalg.norm(coeffs[m] - coeffs_prev[m]) for m in months)
        if Print_Iterations:
            print(f" Iteration {iteration + 1}, delta = {delta:.6f}")
        if delta < tol:
            print(f" Converged after {iteration + 1} iterations.")
            break

    os.makedirs("policy_adp_npy", exist_ok=True)
    for m, P in policy.items():
        np.save(f"policy_adp_npy/policy_{m}.npy", P)

    return policy, coeffs
