# Readme file:


# Objective:

Maximizing energy output over time.

# Assumptions:
- Due to the absence of the energy production formula in the paper, I chose to implement it as follows:

**&Phi;<sub>t</sub>(runoff<sub>t</sub>,Z<sub>t</sub><sup>b</sup>, Z<sub>t</sub><sup>e</sup>) = efficiency*[ log(runoff<sub>t</sub>) + log( max(Z<sub>t</sub><sup>b</sup> - Z<sub>t</sub><sup>e</sup> , 0) ) ]*

# Instructions:

# How to Run

This project uses Python 3.9+.

# Dependencies: 

If you don't already have the required packages installed, run:

`pip install -r requirements.txt`:

# Clean-up:

- Make sure to first remove any existing files by running: 
 - `rm -rf policy_npy` (To remove the data from running Exact DP algorithm)
 - `rm -rf policy_adp_npy` (To remove the data from running Approximate DP Algorithm)

## Test Exact DP:

- To run the code, type into the terminal: `python3 -m Run.runExactDP` (This will take a few minutes, you will see on the console the iterations print alongside the delta values)
- Once it is done, to visualize the data, type into the terminal: `python3 -m SDP.ExactDP_visual`

## Test Approximate DP:

- To run the code, type into the terminal: `python3 -m Run.runApprDP` (This will be fast, you will see on the console the iterations print alongside the delta values)
- Once it is done, to visualize the data, type into the terminal: `python3 -m ADP.ApprDP_visual`

## Personalisation:

- You can change the conversion rate to smaller or higher values by changing the `tol=0.01` value in the parameters of sdp_dam or adp_dam depending on what you wish to test.

- In the `ApprDP` algorathim, you can customize the parameters depending on the precision needed:           (n_zb_samples=90, n_qr_samples=5, n_release_samples=5)

- IMPORTANT!: You can uncomment the code after the comment "# Print Release policy per Initial water level per month:" in both `Run.runExactDP` and `Run.runApprDP` files to see printed to the console the release policy per every inital water level, for every month.

It will look like this: 

Month: April
  - Zb = 740.0 → Avg Release = 0.0
  - Zb = 741.0 → Avg Release = 1.0
  - Zb = 742.0 → Avg Release = 2.0
  - Zb = 743.0 → Avg Release = 3.0
  - Zb = 744.0 → Avg Release = 3.9
  - Zb = 745.0 → Avg Release = 4.9
  - Zb = 746.0 → Avg Release = 5.8
 .....