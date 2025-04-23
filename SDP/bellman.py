
import numpy as np
from runoff.transition_matrices import transition_matrices
from runoff.runoff_data import runoff_qr
from utils import *



def sdp_dam(max_iter=1000, tol=1e-3, verbose=True):
  # Initialize value and policy tables
  J = {month: np.zeros(len(ZB_LEVELS)) for month in months}
  policy = {month: np.zeros(len(ZB_LEVELS)) for month in months}

  for iteration in range(max_iter):
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
              #E(Power) calculated here 
              expected_value += prob * (reward + future_value)

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
