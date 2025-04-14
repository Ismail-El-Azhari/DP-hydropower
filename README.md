# DP-Hydropower

# Objective function:

**F = max { Φₜ + Eₜ₊₁ }**

# Assumptions:
- Due to the absence of the energy production formula in the paper, I chose to implement it as follows:

**&Phi;<sub>t</sub> = (runoff<sub>t</sub> + Z<sub>t</sub><sup>b</sup> - Z<sub>t</sub><sup>e</sup>)*0.9**

Or we can write it as:

**&Phi;<sub>t</sub> = (q<sub>t</sub> + Z<sub>t</sub><sup>b</sup> - Z<sub>t</sub><sup>e</sup>)*0.9**