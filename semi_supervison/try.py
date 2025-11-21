import numpy as np
from scipy.optimize import minimize
import concurrent.futures
from tqdm import tqdm
from scipy.signal import cont2discrete
import json

# average time improvement 
data = [104+233.76,190.38+248.02,423.37+234.44,
        120.74+245.86,234.55+232.99,466.81+237.80,
        118.96+235.51,244.43+235.31,465.19+236.60]

BenchMark = 49286.05+222.61
data = np.array(data)
ave_ratio = np.mean(BenchMark/data)
print("average time improvement:",ave_ratio)


# Average Tracking Performance
BenchMark = 74.73
data = [21.90, 13.71, 12.27, 18.10, 13.74, 12.15, 12.38, 11.95, 12.06,
        14.24, 12.27, 12.53, 13.62, 12.15, 12.88, 12.60, 12.06, 11.90,
        13.35, 12.10, 12.15, 12.12, 12.06, 11.86, 14.50, 13.86, 12.89, 
        12.44, 11.98, 11.43, 13.23, 12.40, 13.91, 13.13, 11.68, 11.64]
result = BenchMark/np.array(data)
print("average tracking performance:",np.mean(result))
# Average Safety Performance