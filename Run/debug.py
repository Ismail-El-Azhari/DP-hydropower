# import numpy as np
# from SDP.utils import months
# from runoff.transition_matrices import transition_matrices

# for i in range(len(months)):
#     curr_month = (i + 6) % 12 or 12
#     next_month = (curr_month % 12) + 1
#     key = f"{curr_month}_{next_month}"

#     P = transition_matrices.get(key)
#     if P is None:
#         print(f" Missing matrix for {key}")
#         continue

#     if P.shape != (10, 10):
#         print(f" Matrix {key} has shape {P.shape} instead of (10, 10)")

#     for row_idx, row in enumerate(P):
#         row_sum = np.sum(row)
#         if not np.isclose(row_sum, 1.0, atol=1e-4):
#             print(f" Row {row_idx} in {key} sums to {row_sum:.4f} (not 1.0)")



# import numpy as np

# policy = np.load("policy2_npy/policy_June_qr.npy")
# for i, row in enumerate(policy):
#     print(f"QR{i+1}: unique values in release policy → {np.unique(row)}")



# import numpy as np

# # Load one full month policy array (shape: 10, len(ZB_LEVELS))
# month = "June"
# policy = np.load(f"policy2_npy/policy_{month}_qr.npy")

# print(f"Shape of policy[{month}] =", policy.shape)

# # Print differences across QR bins for a few zb values
# print(f"\nComparison of releases at selected zb indices:")
# for zb_idx in [0, 20, 40, 60, 80]:  # adjust as needed
#     releases = policy[:, zb_idx]
#     print(f"zb = {740 + zb_idx}: {releases.round(1)}")


# import numpy as np

# policy = np.load("policy2_npy/policy_October_qr.npy")
# zb_values = [750, 760, 770, 780, 790, 800]

# print("Zb-level differences for each QR in October:")
# for zb in zb_values:
#     idx = zb - 740
#     releases = policy[:, idx]
#     print(f"Zb = {zb} → {releases}")


