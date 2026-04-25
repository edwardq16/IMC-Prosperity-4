import numpy as np
from scipy.stats import norm

def bs_call(S, K, T, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)

    d1 = (np.log(S / K) + 0.5 * (sigma ** 2) * T)/(sigma * np.sqrt(T))
    d2 = d1 - (sigma * np.sqrt(T))
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    C = (S * N_d1) - (K * N_d2)

    return C

def bs_delta(S, K, T, sigma):
    if T <= 0 or sigma <= 0:
        if S > K:
            return 1.0
        else:
            return 0.0

    d1 = (np.log(S / K) + 0.5 * (sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def implied_vol(C_mkt, S, K, T, tolerance=1e-5, max_iterations=100):
    if T <= 0 or C_mkt < max(S - K, 0) or C_mkt > S:
        return None

    low = 1e-4
    high = 5.0
    mid = 0.5 * (low + high)

    for i in range(max_iterations):
        mid = 0.5 * (low + high)
        price = bs_call(S, K, T, mid)
        if abs(price - C_mkt) < tolerance:
            return mid
        if price < C_mkt:
            low = mid
        if price > C_mkt:
            high = mid

    return mid

# test cases
print(implied_vol(100, 5250, 5200, 5.0))
print(implied_vol(270, 5250, 5000, 5.0))   # VEV 5000
print(implied_vol(180, 5250, 5100, 5.0))   # VEV 5100
print(implied_vol(50,  5250, 5300, 5.0))   # VEV 5300
print(implied_vol(16,  5250, 5400, 5.0))   # VEV 5400