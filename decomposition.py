from scipy.integrate import quad, dblquad, tplquad, nquad
import numpy as np
 
 
def N_r(K):
    return np.exp(-K**2/2)/(2*np.pi)**(1/2)
 
def mt(t):
    return (np.log(K/S0*np.exp(r*T))+sigma**2*t/2)/sigma*t**0.5
 
def Rt(t):
    return (np.log(K/S0)-(r-sigma**2/2)*t)/sigma*t**0.5
 
def Int_F1(t):
    return 1/t**(1/2)*np.exp(-mt(t)**2/2)/(2*np.pi)**(1/2)
 
def Int_F2(t,u):
    return np.exp(-r*t)*np.exp(-u**2/2)/(2*np.pi)**(1/2)
 
def Int_F3(t):
    return np.exp(-r*t)*N_r(Rt(t))/sigma*t**0.5
 
def S3_put(S0,K,T,r,sigma):
    a = np.max([0, K*np.exp(-r*T)-S0])
    b = np.exp(-r*T)*(sigma*K)/2*quad(Int_F1,0,T)[0]
    c = r*K*dblquad(Int_F2,0,T, 0, Rt)[0]
    d = sigma**2*K/2*quad(Int_F3,0,T)[0]
    put = a +b+c-d
    return put

S0=163.62
K=165
r=0.0241
sigma=0.2848
T=4/73
print(S3_put(S0, K, T, r, sigma))
