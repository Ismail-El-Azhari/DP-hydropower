from SDP.bellman import *
from SDP.utils import *

# Run the SDP solver
policy, J = sdp_dam()

# Display the full optimal policy for each month and each water level
print("\n Optimal Release Policy:")
for month in months:
    print(f"\nMonth: {month}")
    for zb_i, zb in enumerate(ZB_LEVELS):
        release = policy[month][zb_i]
        print(f"  - Zb = {zb:.1f} → Release = {release:.1f}")

# Optionally, you can also show the expected energy
print("\n Expected Energy-to-Go:")
for month in months:
    print(f"\nMonth: {month}")
    for zb_i, zb in enumerate(ZB_LEVELS):
        energy = J[month][zb_i]
        print(f"  - Zb = {zb:.1f} → Energy = {energy:.2f}")


