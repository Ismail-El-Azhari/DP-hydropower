
import numpy as np
from SDP.utils import power,ZB_LEVELS

months = [
    "June", "July", "August", "September", "October", "November",
    "December", "January", "February", "March", "April", "May"
]

# Initialize value and policy tables
J = {month: np.zeros(len(ZB_LEVELS)) for month in months}
policy = {month: np.zeros(len(ZB_LEVELS)) for month in months}
