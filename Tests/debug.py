import numpy as np
from SDP.utils import months
from runoff.transition_matrices import transition_matrices

for i in range(len(months)):
    curr_month = (i + 6) % 12 or 12
    next_month = (curr_month % 12) + 1
    key = f"{curr_month}_{next_month}"

    P = transition_matrices.get(key)
    if P is None:
        print(f"❌ Missing matrix for {key}")
        continue

    if P.shape != (10, 10):
        print(f"⚠️ Matrix {key} has shape {P.shape} instead of (10, 10)")

    for row_idx, row in enumerate(P):
        row_sum = np.sum(row)
        if not np.isclose(row_sum, 1.0, atol=1e-4):
            print(f"⚠️ Row {row_idx} in {key} sums to {row_sum:.4f} (not 1.0)")
