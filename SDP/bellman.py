
import numpy as np
from runoff.transition_matrices import transition_matrices
from runoff.runoff_data import runoff_qr
from SDP.utils import *



def sdp_dam(max_iter=100, tol=1e-3, verbose=True):
  # Initialize value and policy tables
  J = {month: np.zeros(len(ZB_LEVELS)) for month in months}
  policy = {month: np.zeros(len(ZB_LEVELS)) for month in months}
  # I chose beta=0.9 for discounting
  beta=0.9

  for iteration in range(max_iter):
    #Create a copy of the runoff values to be able to compare how they changed later on in my max_diff
    J_prev = {m: J[m].copy() for m in months}
    max_diff = 0

    # Reverse loop over months (backward DP)
    for i in reversed(range(len(months))):
      month = months[i]
      next_month = months[(i + 1) % 12]  # wrap around

      # Key for transition matrix (June = 6)
      curr_month = (i + 6) % 12 or 12
      next_month_num = (curr_month % 12) + 1
      key = f"{curr_month}_{next_month_num}"

      P = transition_matrices[key]
      # if P.shape != (10, 10):
      #   print(f"Issue for {key}: {P.shape}")
      #   print(P)

      runoff_values = runoff_qr[month]

      for zb_i, zb in enumerate(ZB_LEVELS):
        best_value = -np.inf
        best_release = 0

        for release in RELEASES:
          ze = zb - release
          if ze < Z_MIN:
            continue

          expected_value = 0
          
          for qr_idx1 in range(10):
            #The runoff from this level
            runoff = runoff_values[qr_idx1]
            #Calculate the reward based on current runoff and release decision
            reward = power(runoff, zb, ze)
            #Using transition probabilities to get the Expected runoff for next period
            for qr_idx2 in range(10):
              prob = P[qr_idx1][qr_idx2]
              runoff_next = runoff_qr[next_month][qr_idx2]
              #Water level next month
              next_zb = ze + runoff_next

              if next_zb < Z_MIN or next_zb > Z_MAX:
                continue

              next_zb_i = int(round(next_zb)) - Z_MIN
              future_value = J[next_month][next_zb_i]
              # if iteration == 0 and zb == 765 and month == "June":
              #   print(f"[DEBUG] release={release}, runoff={runoff:.2f}, ze={ze}, next_zb={next_zb}, reward={reward:.2f}, future={future_value:.2f}, prob={prob}")

              #E(Power) calculated here, 
              # IMPORTANT: Scaling both reward and value function by 1e3 to keep numerical values small and consistent
              # Assumes all energy values are expressed in kilounits (e.g., kWh), not Wh
              expected_value += prob * (reward + beta*future_value)/1e3
              
          # # Debug trace for a specific condition
          # if zb == 765 and month == "June" and iteration < 2:
          #   print(f"[DEBUG] release={release}, runoff={runoff:.2f}, ze={ze}, "
          #     f"next_zb={next_zb:.2f}, reward={reward:.2f}, "
          #     f"future={future_value:.2f}, expected={expected_value:.2f}")
            
          if expected_value > best_value:
            best_value = expected_value
            best_release = release

        J[month][zb_i] = best_value
        policy[month][zb_i] = best_release

      # After all months and levels, check convergence
    max_diff = max(np.max(np.abs(J[m] - J_prev[m])) for m in months)

    if verbose:
      print(f" Iteration {iteration+1}, Delta = {max_diff:.5f}")
    if max_diff < tol:
      print(f" Converged after {iteration+1} iterations.")
      break
  return policy, J
