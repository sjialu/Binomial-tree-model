import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import quad
mpl.rcParams['font.family'] = 'serif'

def american_option_value(S0, K, T, r, sigma, otype, M):
    # i generate binomial tree
    dt = T / M  # length of time interval
    #df = math.exp(-r * dt)  # discount per interval
    df=1/(1+r*dt)
    #inf = math.exp(r * dt)  # discount per interval
    inf=1+r*dt

    # calculate udp
    u = math.exp(sigma * math.sqrt(dt))  # up movement
    d = 1 / u  # down movement
    #q = (math.exp(r * dt) - d) / (u - d)  # martingale branch probability
    q=(1+r*dt-d)/(u-d)
    
    # initial matrix
    mu = np.arange(M + 1)
    mu = np.resize(mu, (M + 1, M + 1))
    md = np.transpose(mu)
    
    # Calculate the stock price for a one-way movement of a node
    mus = u ** (mu - md)
    mds = d ** md
    
    # Get the stock price at each node
    S = S0 * mus * mds 
        
    # ii. Calculating the expected price of the stock at each node
    mes = S0 * inf ** mu

    # iii. Obtaining the option value of a leaf node
    if otype == 'call':
        V = np.maximum(S - K, 0)     
        # Calculate the gain from early exercise of each node
        oreturn = mes - K
    else:
        V = np.maximum(K - S, 0)       
        oreturn = K - mes

    # iv. Comparison of the proceeds of progressive forward weighted average
    # discount and early exercise of options to obtain the initial option value in the period
    for z in range(0, M):  # backwards iteration
        #Calculation of late prices for late discounting
        ovalue = (q * V[0:M - z, M - z] +
                         (1 - q) * V[1:M - z + 1, M - z]) * df
        #Update the option value column by column,
        #Option prices are taken as the maximum of later discounted and early exercise gains
        V[0:M - z, M - z - 1] = np.maximum(ovalue, oreturn[0:M - z, M - z - 1])
        
    return V[0, 0]

S0 = 133.6  # index level
K = 133  # option strike
T =2/365# maturity date
r = 0.0231#Update the option value column by column, equivalent to discounting forward level by level in a binomial tree
        #Option prices are taken as the maximum of later discounted and early exercise gains # risk-less short rate
sigma = 0.3818 # volatility
print(american_option_value(S0, K, T, r, sigma, 'put', 1000))

