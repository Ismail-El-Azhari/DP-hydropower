from SDP.ExactDP2 import *
from SDP.utils import *

# Run the SDP solver
policy, J = sdp_dam(debug=True)  # this returns the standard policy (1D)

# Load the full QR-level policy for visualization/inspection
import numpy as np
policy_by_qr = {
    m: np.load(f"policy2_npy/policy_{m}_qr.npy")  # shape: (10, len(ZB_LEVELS))
    for m in months
}

# Print the average optimal policy per Zb for each month
# for month in months:
#     print(f"\nMonth: {month}")
#     for i, zb in enumerate(ZB_LEVELS):
#         release = np.mean(policy_by_qr[month][:, i])
#         print(f"  - Zb = {zb:.1f} → Avg Release = {release:.1f}")
