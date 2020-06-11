import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sts
from statsmodels.distributions.empirical_distribution import ECDF

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
    return (np.log(x) - 1.0 ) / 1.0


A = 1.0
B = 2.0
y0 = 3.0
N=30

M = 10  # int(np.sqrt(n)) if n <= 100 else int(4 * np.log10(n)) # эмпирически определяем количество интервалов
m = N // M

Y = random_numbers_sample(N, A, B)

Y.sort()
table = pd.DataFrame(data={"Значение": Y})
table =table.transpose()
print(table)

ecdf = ECDF(Y)

plt.step(ecdf.x, ecdf.y, label='Эмпирическая F(Y)')

xx = np.linspace(np.exp(1.0), np.exp(2.0), 100)
yy = F(xx)
plt.plot(xx, yy, label='Теоретическая F(Y)')

plt.ylabel('$F(y)$')
plt.xlabel('$y$')
plt.legend(loc='upper left')
plt.show()

dp = []
dm = []

for i in range(N - 1):
    dp.append(abs((i + 1) / N - F(Y[i])))
    dm.append(abs(i / N - F(Y[i])))

d = max(max(dm), max(dp))
print("\nd=",d)

l = d * np.sqrt(N)
print("l=",l)

if l < 1.36 and l > 0.0:
    print("нет оснований отвергать гипотезу H0")
else:
    print("есть основания отвергнуть гипотезу H0")