import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sts

# m=1; s=2*(3^0.5)

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


def hist_eq_prob(sample, m, M):
    sample.sort()
    A = [sample[0]]
    B = []
    for i in range(1, M):
        A.append((sample[i * m - 1]+sample[i * m])/2)
        B.append((sample[i * m - 1]+sample[i * m])/2)
    B.append(sample[-1])
    h = []
    for i in range(len(A)):
        h.append(B[i] - A[i])
    y = []
    x = []
    s = A[0]
    for hi in h:
        y.append(1.0 / (M * hi))
        x.append(s)
        s += hi
    x.append(s)
    return (A, B, h, x, y)


A = 1.0
B =2.0
y0 = 3.0
N=200

M = 10  # int(np.sqrt(n)) if n <= 100 else int(4 * np.log10(n)) # эмпирически определяем количество интервалов
m = N // M

Y = random_numbers_sample(N, A, B)

A, B, h, x, y = hist_eq_prob(Y, m, M)
yg=y.copy()
yg.append(y[-1])
#plt.fill_between(x, y1=y, y2=[0] * len(y), color='b', step='post', alpha=0.5, label="Равновероятностная гистограмма")
plt.fill_between(x, y1=yg, color='b', step='post', alpha=0.5, label="Равновероятностная гистограмма")

xx = np.linspace(np.exp(1.0), np.exp(2.0))
py = 1 / xx
plt.plot(xx, py, label='Теоретическая f(Y)', c='r')

plt.legend(loc='upper left')
plt.xlabel("y")
plt.ylabel("f(Y)")
plt.show()

table = pd.DataFrame(data={"A_i" : A, "B_i": B, "h_i": h, "m_i": [m]*len(h), "f_i": y})
print(table)

p = []
p_ = [m / N] * len(A)
FB = []
FA = []

for i in range(len(A)):
    FB.append(1.0/1.0 * np.log(B[i]))
    FA.append(1.0/1.0 * np.log(A[i]))
    p.append(FB[i]-FA[i])

ksi=[]
for i in range(len(A)):
    ksi.append(N * (p[i] - p_[i])**2 / p[i])

table2 = pd.DataFrame(data={"F(A_i)" : FA, "F(B_i)": FB, "p_i": p, "p_i*": p_, "ksi_i": ksi})
print("\n")
print(table2)

print("\n")
ksi_stat = sum(ksi)
print("ksi_stat=", ksi_stat)

k = M - 1
print("k=",k)

if ksi_stat < 21.07 and ksi_stat > 2.09:
    print("нет оснований отвергать гипотезу H0")
else:
    print("есть основания отвергнуть гипотезу H0")