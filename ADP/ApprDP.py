import numpy as np
from dam_model.transition_matrices import transition_matrices
from dam_model.runoff_data import runoff_qr
from dam_model.utils import *
import os
import random  # For random sampling

def adp_dam(max_iter=100, tol=0.01, Print_Iterations=True, debug=False,
                   n_zb_samples=90,  # Number of ZB states left untoched
                   n_qr_samples=5,    # Number of QR samples for expectation
                   n_release_samples=5): # Number of release actions to sample
    
    ''' This version uses state and action sampling to reduce the computational burden associated 
    with the curse of dimensionality.'''

    '''Comments only on the parts that are different from Exact DP to make the code more readible, the code is identical, I just sampled the Reservoir levels and runoff bins'''

    J = {month: np.zeros(len(ZB_LEVELS)) for month in months}
    policy = {month: np.zeros(len(ZB_LEVELS)) for month in months}
    policy_by_qr = {month: np.zeros((10, len(ZB_LEVELS))) for month in months}  
    beta = 0.9

    for iteration in range(max_iter):
        J_prev = {m: J[m].copy() for m in months}
        for i in reversed(range(len(months))):
            month = months[i]
            next_month = months[(i + 1) % 12]

            curr_num = (i + 6) % 12 or 12
            next_num = (curr_num % 12) + 1
            P = transition_matrices[f"{curr_num}_{next_num}"]

            runoff_values = runoff_qr[month]
            M = len(runoff_values)

            # Randomly sample a subset of ZB (reservoir level) states
            zb_indices = random.sample(range(len(ZB_LEVELS)), min(n_zb_samples, len(ZB_LEVELS)))

            for zb_i in zb_indices:
                zb = ZB_LEVELS[zb_i]
                avg_value = 0.0

                #for index1 in range(M): # changed
                qr_indices = np.linspace(0, M-1, n_qr_samples, dtype=int)
                for index1 in qr_indices:
                    best_value = -np.inf
                    best_release = 0

                    # Sample a few runoff bins (QR samples) instead of using all them
                    sampled_releases = np.linspace(min(RELEASES), max(RELEASES), n_release_samples)

                    for release in sampled_releases:
                        ze = zb - release
                        if ze < Z_MIN:
                            continue

                        reward = power(runoff_values[index1], zb, ze)
                        penalty = -1000 if (ze < 750 or ze > 830) else 0

                        expected_value = 0.0
                        for index2 in range(M):
                            prob = P[index1][index2]
                            runoff_next = runoff_qr[next_month][index2]
                            next_zb = ze + runoff_next

                            if not (Z_MIN <= next_zb <= Z_MAX):
                                continue

                            next_zb_i = int(round(next_zb)) - Z_MIN
                            future = J_prev[next_month][next_zb_i]
                            expected_value += prob * (reward + penalty + beta * future)

                        if expected_value > best_value:
                            best_value = expected_value
                            best_release = release

                    policy_by_qr[month][index1, zb_i] = best_release
                    avg_value += best_value

                # After looping all QR bins, store average value and policy, here instead of M, we devide by the number of qr samples we chose
                policy[month][zb_i] = np.mean(policy_by_qr[month][:, zb_i])
                J[month][zb_i] = avg_value / len(qr_indices) 

        # check convergence
        delta = max(np.max(np.abs(J[m] - J_prev[m])) for m in months)
        if Print_Iterations:
            print(f" Iteration {iteration+1}, delta = {delta:.6f}")
        if delta < tol:
            print(f" Converged after {iteration+1} iterations.")
            break

    os.makedirs("policy_adp_npy", exist_ok=True)

    for m, P in policy_by_qr.items():
        np.save(f"policy_adp_npy/policy_{m}_approx.npy", P) 

    return policy, J
