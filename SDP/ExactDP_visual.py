import numpy as np
import matplotlib.pyplot as plt

from SDP.utils import months, ZB_LEVELS, Z_MIN, Z_MAX
from runoff.runoff_data import runoff_qr  # to get QR values per month

# 1) Load the full QR-level policy arrays
policy_by_qr = {
    m: np.load(f"policy2_npy/policy_{m}_qr.npy")  # shape: (10, len(ZB_LEVELS))
    for m in months
}

# 2) Create plot: Start-of-month Zb vs End-of-month Ze = Zb - release for all 10 QR bins
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for ax, m in zip(axes, months):
    for i, qr in enumerate(runoff_qr[m]):
        Zb = ZB_LEVELS
        release = policy_by_qr[m][i]  # shape: (len(ZB_LEVELS),)
        Ze = Zb - release

        ax.plot(Ze, Zb, label=f"QR{i+1}")
    
    ax.set_title(m)
    ax.set_xlabel("End-of-Month Level Ze (m)")
    ax.set_ylabel("Start-of-Month Level Zb (m)")
    ax.set_xlim(Z_MIN, Z_MAX)
    ax.set_ylim(Z_MIN, Z_MAX)
    ax.legend(fontsize="x-small", ncol=2)

plt.tight_layout()
plt.show()
