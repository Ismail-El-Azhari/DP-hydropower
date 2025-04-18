# Here we will simulate the data 

import numpy as np
import pandas as pd
from scipy.stats import lognorm
from runoff.runoff_data import runoff_qr


def simulate_monthly_runoff(n_samples=5000):
    simulated_data = {}

    for month, values in runoff_qr.items():
        
        data = np.array(values)

        shape, loc, scale = lognorm.fit(data, floc=0)

        samples = lognorm.rvs(shape, loc=loc, scale=scale, size=n_samples)

        simulated_data[month] = samples

    return simulated_data

#print(data["August"][:10]) 

if __name__ == "__main__":

    simulated_data = simulate_monthly_runoff(n_samples=5000)
    df = pd.DataFrame(simulated_data)
    df.to_csv("runoff_simulated.csv", index=False)

    print("Simulated runoff saved to runoff_simulated.csv")

def discretize_runoff_to_qr_bins(csv_path, output_csv):
    
    df = pd.read_csv("runoff_simulated.csv")
    qr_bins_df = pd.DataFrame()

    for month in df.columns:
        runoff_values = df[month].values

        quantile_edges = np.quantile(runoff_values, q=np.linspace(0, 1, 11))
        # Assign each value to a bin
        bin_indices = np.digitize(runoff_values, quantile_edges, right=True)
        # Fix overflow: 
        bin_indices[bin_indices > 10] = 10

        qr_bins_df[month] = bin_indices

    qr_bins_df.to_csv(output_csv, index=False)
    print(f" Saved QR bin labels to {output_csv}")
    return qr_bins_df


if __name__ == "__main__":
    # Simulate from log-normal marginals
    simulated_data = simulate_monthly_runoff(n_samples=5000)
    df = pd.DataFrame(simulated_data)
    df.to_csv("runoff_simulated.csv", index=False)
    print("Simulated runoff saved to runoff_simulated.csv")

    # Discretize both files
    discretize_runoff_to_qr_bins("runoff_simulated.csv", "runoff_bins.csv")
    discretize_runoff_to_qr_bins("runoff_gibbs.csv", "runoff_gibbs_bins.csv")

