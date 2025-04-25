import numpy as np 

months = [
    "June", "July", "August", "September", "October", "November",
    "December", "January", "February", "March", "April", "May"
]

# I defined the power function this way:
def power(runoff, zb, ze, efficiency=0.9):
  release = max(zb - ze,0)
  return efficiency * np.log1p(runoff) * np.log1p(release)

# Water levels in meters (states)
#I chose the water evels based on figure 11 in the paper
Z_MIN = 740
Z_MAX = 830
Z_STEP = 1
#Discretizing water levels by 1 intervals
ZB_LEVELS = np.arange(Z_MIN, Z_MAX + 1, Z_STEP)

# Release levels in meters (actions)
# RELEASE_MAX = Z_Max-Z_Min

RELEASE_MIN = 0
RELEASE_MAX = 90
RELEASE_STEP = 1

#Discretizing release levels
RELEASES = np.arange(RELEASE_MIN, RELEASE_MAX + 1, RELEASE_STEP)



