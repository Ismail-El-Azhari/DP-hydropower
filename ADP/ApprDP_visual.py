import numpy as np
import matplotlib.pyplot as plt

from dam_model.utils import months, ZB_LEVELS, Z_MIN, Z_MAX


# Load the full QR-level policy arrays
policy_by_qr = {
    m: np.load(f"policy_adp_npy/policy_{m}_approx.npy")  # shape: (10, len(ZB_LEVELS))
    for m in months
}


# Create plot
fig, axes = plt.subplots(3, 4, figsize=(12, 8))
axes = axes.flatten()

for ax, m in zip(axes, months):
    for i in range(10):
        Zb = ZB_LEVELS
        release = policy_by_qr[m][i]
        Ze = Zb - release

        ax.plot(Ze, Zb, label=f"QR{i+1}", alpha=0.8)

    ax.set_title(m)
    ax.set_xlabel("End-of-Month Level Ze (m)")
    ax.set_ylabel("Start-of-Month Level Zb (m)")
    ax.set_xlim(Z_MIN, Z_MAX)
    ax.set_ylim(Z_MIN, Z_MAX)
    ax.legend(fontsize="x-small", ncol=2)

plt.tight_layout()
plt.show()
