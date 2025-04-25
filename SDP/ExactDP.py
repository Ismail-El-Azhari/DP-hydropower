import numpy as np
from dam_model.transition_matrices import transition_matrices
from dam_model.runoff_data import runoff_qr
from dam_model.utils import *
import os

def sdp_dam(max_iter=100, tol=0.01, Print_Iterations=True, debug=False):

    # Initialize value function and policy dictionaries for each month
    J = {month: np.zeros(len(ZB_LEVELS)) for month in months}
    policy = {month: np.zeros(len(ZB_LEVELS)) for month in months}
    beta = 0.9  # Discount factor

    # This dictionary keeps all policies by runoff bin for visualization (Does not play a role in the code)
    policy_by_qr = {month: np.zeros((10, len(ZB_LEVELS))) for month in months}  
    

    for iteration in range(max_iter):
        # Keep a copy of the value function to check convergence later
        J_prev = {m: J[m].copy() for m in months}

        # Perform a backward dynamic 
        for i in reversed(range(len(months))):
            month = months[i]
            next_month = months[(i + 1) % 12]

            ''' Find the correct transition matrix based on current and next month
            Reminder: i=0->June(6),i=11->May(5), first month is June NOT January '''
            curr_num = (i + 6) % 12 or 12 
            next_num = (curr_num % 12) + 1
            P = transition_matrices[f"{curr_num}_{next_num}"]

            runoff_values = runoff_qr[month]   # list of 10 QR bins
            M = len(runoff_values)

            # On the first iteration, we will start with a fized water level value for June
            if iteration == 0 and month == "June":
                zb_indices = [765 - Z_MIN]
            else:
                zb_indices = range(len(ZB_LEVELS))

            # Loop through every water level
            for zb_i in zb_indices:
                zb = ZB_LEVELS[zb_i]
                avg_value = 0.0
                # Loop through every possible runoff value
                for index1 in range(M):
                    best_value = -np.inf
                    best_release = 0
                    # Try every admissible policy (amount of water to release)
                    for release in RELEASES:
                        ze = zb - release
                        if ze < Z_MIN:
                            continue
                        #Calculate immidiate reward and penalty
                        reward = power(runoff_values[index1], zb, ze)
                        penalty = -1000 if (ze < 750 or ze > 830) else 0

                        expected_value = 0.0
                        # Loop through all the probabilities 
                        for index2 in range(M):
                            prob = P[index1][index2] #Get the p from transition matrix
                            runoff_next = runoff_qr[next_month][index2] 
                            next_zb = ze + runoff_next #next water level=current water level + runoff
                            
                            # Reject if it does not respect the constraints
                            if not (Z_MIN <= next_zb <= Z_MAX):
                                continue
                            
                            next_zb_i = int(round(next_zb)) - Z_MIN
                            future = J_prev[next_month][next_zb_i]
                            expected_value += prob * (reward + penalty + beta * future)

                        # Keep track of the best release decision
                        if expected_value > best_value:
                            best_value = expected_value
                            best_release = release

                    policy_by_qr[month][index1, zb_i] = best_release 
                    avg_value += best_value

                # After trying all QR bins, store average value and corresponsing policy
                policy[month][zb_i] = np.mean(policy_by_qr[month][:, zb_i])
                J[month][zb_i] = avg_value / M

        # Check convergence, here the choice was the max difference
        delta = max(np.max(np.abs(J[m] - J_prev[m])) for m in months)

        # Printing iterations to keep track of convergence
        if Print_Iterations:
            print(f" Iteration {iteration+1}, delta = {delta:.6f}")
        if delta < tol:
            print(f" Converged after {iteration+1} iterations.")
            break

    # Save both main policy and the detailed per-QR policy arrays
    os.makedirs("policy_npy", exist_ok=True)

    for m, P in policy_by_qr.items():
        np.save(f"policy_npy/policy_{m}_qr.npy", P)

    return policy, J
