def power(runoff, zb, ze, efficiency=0.9):
  return (runoff + zb - ze) * efficiency

import numpy as np

# Water levels in meters (states)
Z_MIN = 740
Z_MAX = 830
Z_STEP = 1
ZB_LEVELS = np.arange(Z_MIN, Z_MAX + 1, Z_STEP)

# Release levels in meters (actions)
RELEASE_MIN = 0
RELEASE_MAX = 50
RELEASE_STEP = 1
RELEASES = np.arange(RELEASE_MIN, RELEASE_MAX + 1, RELEASE_STEP)
