import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import scipy.integrate as integr

def random_numbers_sample(n, a, b):
    randomiser = sts.uniform()
    randoms = randomiser.rvs(size=n)
    X = []
    for i in range(n):
        x = randoms[i] * (b - a) + a
        X.append(x)
    Y = []
    for i in range(n):
        y = np.exp(X[i])
        Y.append(y)
    return Y


A = 1.0
B = 2.0
y0 = 3.0
N = 20

Y = random_numbers_sample(N, A, B)

e_M = sum(Y) / N
print("e_M=", e_M)

def empirical_D(Y, n, M):
    e_D = 0.0
    for i in range(N):
        e_D = e_D + (Y[i]-M)**2
    e_D = e_D / (n-1)
    return e_D

e_D = empirical_D(Y, N, e_M)
print("e_D=", e_D)

gammas = [0.9, 0.95, 0.98, 0.99]
chi2_rv = sts.chi2(19)
arr = chi2_rv.rvs(100000)
chip = []
chim = []
for i in gammas:
    tmp = sts.mstats.mquantiles(arr, prob=[(1-i)/2, (1+i)/2])
    chip.append(tmp[0])
    chim.append(tmp[1])

intervals = []
for i in range(len(chim)):
    intervals.append((N * e_D / chim[i], N * e_D / chip[i]))
print(intervals)
lens=[i[1] - i[0] for i in intervals]

plt.plot(gammas, lens, label ="Эмпирическое матожидание")
plt.xlabel("$Значимотсь$")
plt.ylabel("Величина итервалла")
plt.legend(loc='upper left')
plt.show()

t_M = integr.quad(lambda x: (x * 1.0 / x), np.exp(1.0), np.exp(2.0))[0]
print("t_M=", t_M)

t_intervals = []
for i in range(len(chim)):
    t_intervals.append((N * empirical_D(Y, N, t_M) / chim[i], N * empirical_D(Y, N, t_M) / chip[i]))
print(t_intervals)
t_lens=[i[1] - i[0] for i in t_intervals]

plt.plot(gammas, t_lens, label ="Теоретическое матожидание")
plt.xlabel("$Значимотсь$")
plt.ylabel("Величина итервалла")
plt.legend(loc='upper left')
plt.show()

plt.plot(gammas, lens, label="Эмпирическое матожидание")
plt.plot(gammas, t_lens, label="Теоретическое матожидание")
plt.xlabel("Значимотсь")
plt.ylabel("Величина итервалла")
plt.legend(loc='upper left')
plt.show()

def inter_N(n):
    Y = random_numbers_sample(n, A, B)
    e_M = sum(Y) / n
    gammas = [0.9, 0.95, 0.98, 0.99]
    chi2_rv = sts.chi2(n - 1)
    arr = chi2_rv.rvs(100000)
    chip = []
    chim = []
    for i in gammas:
        tmp = sts.mstats.mquantiles(arr, prob=[(1 - i) / 2, (1 + i) / 2])
        chip.append(tmp[0])
        chim.append(tmp[1])

    intervals = []
    for i in range(len(chim)):
        intervals.append((n * empirical_D(Y, n, e_M) / chim[i], n * empirical_D(Y, n, e_M) / chip[i]))

    t_intervals = []
    for i in range(len(chim)):
        t_intervals.append((n * empirical_D(Y, n, t_M) / chim[i], n * empirical_D(Y, n, t_M) / chip[i]))

    lens = [i[1] - i[0] for i in intervals]
    t_lens = [i[1] - i[0] for i in t_intervals]

    plt.plot(gammas, lens, label="Эмпирическое матожидание")
    plt.plot(gammas, t_lens, label="Теоретическое матожидание")
    plt.xlabel("Значимотсь (n=" + str(n) + ")")
    plt.ylabel("Величина итервалла")
    plt.legend(loc='upper left')
    plt.show()
    return lens, t_lens

#для gamma = 0.99

Ns = [30, 50, 70, 100, 150]
ints = []
t_ints=[]
for i in Ns:
    ints_pair = inter_N(i)
    ints.append(ints_pair[0][-1])
    t_ints.append(ints_pair[1][-1])


plt.plot(Ns, ints)
plt.xlabel("Размер выборки")
plt.ylabel("Размер интервалла(эмипирически)")
plt.show()

plt.plot(Ns, t_ints)
plt.xlabel("Размер выборки")
plt.ylabel("Размер интервалла(теоретически)")
plt.show()