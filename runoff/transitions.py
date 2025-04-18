import pandas as pd
import numpy as np

def compute_transition_matrices(csv_path):
    df = pd.read_csv(csv_path)
    months = list(df.columns)
    transition_matrices = {}

    for t in range(len(months) - 1):
        m1, m2 = months[t], months[t + 1]

        # Create a 10x10 matrix
        matrix = np.zeros((10, 10))

        for i in range(len(df)):
            from_bin = df.loc[i, m1] - 1  # QR1 = index 0
            to_bin = df.loc[i, m2] - 1
            matrix[from_bin, to_bin] += 1

        # Normalize each row
        row_sums = matrix.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            matrix = np.divide(matrix, row_sums, where=row_sums != 0)

        transition_matrices[f"{m1}_to_{m2}"] = matrix

    return transition_matrices


if __name__ == "__main__":
    matrices = compute_transition_matrices("runoff_gibbs_bins.csv")
    print("Transition matrix from June to July:")
    print(np.round(matrices["June_to_July"], 3))

