
import numpy as np
from runoff.transition_matrices import transition_matrices
from runoff.runoff_data import runoff_qr
from SDP.utils import *



def sdp_dam(max_iter=100, tol=1e-5, Print_Iterations=True, debug=False):
  # Initialize value and policy tables, all 
  J = {month: np.zeros(len(ZB_LEVELS)) for month in months}
  policy = {month: np.zeros(len(ZB_LEVELS)) for month in months}
  # I chose beta=0.9 for discounting
  beta=0.9

  for iteration in range(max_iter):
    #Create a copy of the runoff values to be able to compare how they changed later on in my max_diff
    J_prev = {month: J[month].copy() for month in months}
    max_diff = 0

    # Reverse loop over months (backward DP)
    for i in reversed(range(len(months))):
      month = months[i]
      next_month = months[(i + 1) % 12]  # wrap around

      # Key for transition matrix 
      # months start at June, Examples: i=0->June(6), i=7-> January(1), i=11->May(5)
      curr_month = (i + 6) % 12 or 12
      next_month_num = (curr_month % 12) + 1
      key = f"{curr_month}_{next_month_num}"

      P = transition_matrices[key]
      
      runoff_values = runoff_qr[month] #10 QR levels for that month
      # Restrict to zb = 765 only in the first iteration for June
      if iteration == 0 and month == "June":
        zb_indices = [int(765) - Z_MIN]
      else:
        zb_indices = range(len(ZB_LEVELS))

      tracked_zbs = [765, 790, 820]

      for zb_i in zb_indices:
        zb=ZB_LEVELS[zb_i]
        best_value = -np.inf
        best_release = 0

        for release in RELEASES:
          ze = zb - release
          if ze < Z_MIN: #Can NOT release so much water that i go below Z_MIN
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

              if next_zb < Z_MIN or next_zb > Z_MAX: #Option rejected if it doesn't satisfy restrictions
                continue

              next_zb_i = int(round(next_zb)) - Z_MIN
              if 0 <= next_zb_i < len(ZB_LEVELS):
                future_value = J[next_month][next_zb_i]
              else:
                  continue


              penalty = -0.1 * abs(750 - ze) ** 2 if ze < 750 or ze > 830 else 0

              expected_value += prob * (reward +penalty+ beta*future_value)/1e3
            
          if expected_value > best_value:
            best_value = expected_value
            best_release = release

        if debug and month == "August" and zb in [765, 790, 820] and release in [10, 20, 30]:
          print(f"[DEBUG] zb={zb}, release={release}, reward={reward:.4f}")


        J[month][zb_i] = best_value
        policy[month][zb_i] = best_release

      # After all months and levels, check convergence
    max_diff = max(np.max(np.abs(J[m] - J_prev[m])) for m in months)


    if Print_Iterations:
      print(f" Iteration {iteration+1}, Delta = {max_diff:.5f}")
    if max_diff < tol:
      print(f" Converged after {iteration+1} iterations.")
      break
  return policy, J
