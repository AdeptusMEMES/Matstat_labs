import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sts

def random_numbers_sample(n, a, b):
    randomiser = sts.uniform()
    randoms = randomiser.rvs(size=n)
    X = []
    for i in range(n):
        x = randoms[i] * (b - a) + a
        X.append(x)
    Y = []
    for i in range(n):
        y =  np.exp(X[i])
        Y.append(y)
    return Y

def F(x):
    return np.log(x) - 1.0

A = 1.0
B =2.0
y0 = 3.0
N=50

Y = random_numbers_sample(N, A, B)
Y.sort()

Fny = []
Fy = []
delta = []

for i in range(N):
    Fny.append((i + 0.5) / N)
    Fy.append(F(Y[i]))
    delta.append((Fny[i] - Fy[i])**2)

table = pd.DataFrame(data={"y": Y, "F_n": Fny, "F": Fy, "delta": delta})
print(table)

stat_mis = 1. / (12 * N) + sum(delta)
print("\n stat_mis=", stat_mis)

if stat_mis < 0.461:
    print("нет оснований отвергать гипотезу H0")
else:
    print("есть основания отвергнуть гипотезу H0")