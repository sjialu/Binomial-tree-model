import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import quad

def sample_paths(r, sigma, S_0, T, M, N):
    '''
    r: risk free rate
    sigma: volatility
    S_0: initial stock price
    T: expire date
    M: number of segments in which the share price change process is discrete
    N: number of paths
    '''
    data = S_0*np.ones((N,1))
    for i in range(M):
        normal_variables = np.random.normal(0,1,(N,1))
        ratios = np.exp((r-0.5*sigma*sigma)*T/M +normal_variables*sigma*(T/M)**0.5)
        # data[:,-1:]he shape of the intercepted data dimension can be kept as(N,1)
        data = np.concatenate((data,data[:,-1:]*ratios),axis=1)
    return data

def linear_fitting(X,Y):
    X = np.array(X)
    Y = np.array(Y)
    S0,S1,S2,S3,S4 = len(X),sum(X),sum(X*X),sum(X**3),sum(X**4)
    V0,V1,V2= sum(Y), sum(Y*X), sum(Y*X*X)
    coeff_mat = np.array([[S0,S1,S2],[S1,S2,S3],[S2,S3,S4]])
    target_vec = np.array([V0,V1,V2])
    inv_coeff_mat = np.linalg.inv(coeff_mat)
    fitted_coeff = np.matmul(inv_coeff_mat,target_vec)
    resulted_Ys = fitted_coeff[0]+fitted_coeff[1]*X+fitted_coeff[2]*X*X
    #print(fitted_coeff)
    return resulted_Ys
def MC_American_put_price(S_0,K,T,r,sigma,M,N):
    # data (paths_num,steps_num+1)。
    data = sample_paths(r,sigma,S_0,T,M,N)
    #print(data)
    option_prices = np.maximum(K-data[:,-1], 0)
    for i in range(M-1,0,-1):
        # The option price is discounted to the current moment.
        option_prices *= 1/(1+r*T/M)#np.exp(-r*T/M)
        # Linear regression fit to update option prices.
        option_prices = linear_fitting(data[:,i], option_prices)
        # Determine if the option should be executed and update the price.
        option_prices = np.maximum(option_prices,K-data[:,i])
    # The recursion back is the next moment from the initial moment,
    # so it needs to be discounted once more.
    option_prices *= 1/(1+r*T/M)#np.exp(-r*T/M)
    return np.average(option_prices)


S_0 = 133.6  # index level
K = 120 # option strike
T = 11/12 # maturity date
r = 0.0409 # LIBOR rate
sigma = 0.4042 # volatility
p = MC_American_put_price(S_0,K,T,r,sigma,500,2000)
print(p)
