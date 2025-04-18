import numpy as np
import pandas as pd
from copulas.bivariate import Clayton, Frank, Gumbel
from scipy.stats import rankdata
from scipy.stats import lognorm


def fit_copula_for_june_july():
    # Step 1: Load the simulated runoff data
    df = pd.read_csv("runoff_simulated.csv")  # file is in root

    # Step 2: Extract June and July values
    june = df["June"].values
    july = df["July"].values

    # Step 3: Convert to uniform [0, 1] using rank transformation
    u = rankdata(june, method='average') / (len(june) + 1)
    v = rankdata(july, method='average') / (len(july) + 1)
    data = np.column_stack([u, v])

    # Step 4: Fit copula
    model = Frank()
    model.fit(data)

    print("Fitted Frank copula from 5000 samples.")
    print("θ =", model.theta)

    # Step 5: Sample some new (u,v) values from the copula
    samples = model.sample(10)
    print("🔢 Sampled (u,v) pairs:")
    print(samples)


if __name__ == "__main__":
    fit_copula_for_june_july()


def fit_all_copulas_and_marginals(csv_path="runoff_simulated.csv"):
    df = pd.read_csv(csv_path)
    months = list(df.columns)

    copulas = {}
    marginals = {}

    for i in range(len(months)):
        # Fit log-normal marginal to month i
        values = df[months[i]].values
        shape, loc, scale = lognorm.fit(values, floc=0)
        marginals[months[i]] = (shape, loc, scale)

        if i < len(months) - 1:
            # Fit Frank copula to (month_i, month_{i+1})
            u = rankdata(df[months[i]], method="average") / (len(df) + 1)
            v = rankdata(df[months[i + 1]], method="average") / (len(df) + 1)
            model = Frank()
            model.fit(np.column_stack([u, v]))
            copulas[f"{months[i]}_to_{months[i+1]}"] = model

    return copulas, marginals, months    
