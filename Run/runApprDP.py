from ADP.ApprDP import *
from dam_model.utils import *

# Run the SDP solver
policy, J = adp_dam()


# Print Release policy per Initial water level per month:

# # Load the full QR-level policy for visualization/inspection

# import numpy as np
# policy_by_qr = {
#     m: np.load(f"policy_adp_npy/policy_{m}_approx.npy")  # shape: (10, len(ZB_LEVELS))
#     for m in months
# }

# # Print the average optimal policy per Zb for each month

# for month in months:
#     print(f"\nMonth: {month}")
#     for i, zb in enumerate(ZB_LEVELS):
#         release = np.mean(policy_by_qr[month][:, i])
#         print(f"  - Zb = {zb:.1f} → Avg Release = {release:.1f}")
