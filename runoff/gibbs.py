from scipy.stats import lognorm
from runoff.copulas import fit_all_copulas_and_marginals
import numpy as np
import pandas as pd

def gibbs_sample_one_sequence(copulas, marginals, months):
    u_sequence = []
    runoff_sequence = []

    # Step 1: initialize u for the first month (June)
    u = np.random.rand()
    u_sequence.append(u)

    # Step 2: sample u_{t+1} from copula, given u_t
    for i in range(len(months) - 1):
        key = f"{months[i]}_to_{months[i+1]}"
        copula = copulas[key]

        # Conditional sampling using inverse transform:
        # Sample v ∈ [0,1], then fix u, and get conditional u_next
        v = np.random.rand()
        u = copula.percent_point([v], [u])[0]  # gives u_{t+1} given u_t
        u_sequence.append(u)

    # Step 3: transform all u_t to runoff using inverse log-normal CDF
    for i, u in enumerate(u_sequence):
        shape, loc, scale = marginals[months[i]]
        runoff = lognorm.ppf(u, shape, loc=loc, scale=scale)
        runoff_sequence.append(runoff)

    return dict(zip(months, runoff_sequence))

def gibbs_sample_sequences(n, copulas, marginals, months):
    all_sequences = []

    for _ in range(n):
        sequence = gibbs_sample_one_sequence(copulas, marginals, months)
        all_sequences.append(sequence)

    df = pd.DataFrame(all_sequences)
    df.to_csv("runoff_gibbs.csv", index=False)
    print(f"Saved {n} Gibbs-sampled sequences to runoff_gibbs.csv")

    return df



if __name__ == "__main__":
    copulas, marginals, months = fit_all_copulas_and_marginals()
    
    # Generate 5000 sequences
    df = gibbs_sample_sequences(5000, copulas, marginals, months)

    # Show first few lines
    print("\n Sample of Gibbs-generated runoff:")
    print(df.head())

