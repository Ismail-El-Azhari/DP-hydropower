import numpy as np
from SDP.utils import *
from runoff.transition_matrices import transition_matrices
from runoff.runoff_data import runoff_qr


# numbers=np.array([i*10 for i in range(10)])
# print(numbers)

# listy=np.arange(2,10,2)
# print(listy)

# J={month: np.zeros(len(ZB_LEVELS)) for month in months}
# print(J)

# for i in range(12):
   
#    print(f"{i%13 or 12}_{(i+1)%13 or 12}") 

# for i in range(12):
#   curr_month = (i + 6) % 12 or 12
#   next_month = (curr_month % 12) + 1
#   key = f"{curr_month}_{next_month}"
#   print(key)

# for i in reversed(range(len(months))):
#   curr_month = (i + 6) % 12 or 12
#   next_month_num = (curr_month % 12) + 1
#   key = f"{curr_month}_{next_month_num}"
#   print(key)


