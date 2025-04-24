from SDP.bellman import *
from SDP.utils import *

# Run the SDP solver
policy, J = sdp_dam(debug=True)


# Print the optimal policy for each month
# for month in months:
#     print(f"\nMonth: {month}")
#     for i, zb in enumerate(ZB_LEVELS):
#         release = policy[month][i]
#         print(f"  - Zb = {zb:.1f} → Release = {release:.1f}")



