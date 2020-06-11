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

e_D = 0.0
for i in range(N):
    e_D += (Y[i]-e_M)**2
e_D = e_D / (N-1)
print("e_D=", e_D)

#n - 1 = 19
gammas = [0.9, 0.95, 0.98, 0.99]
t_rv = sts.t(19)
arr = t_rv.rvs(1000000)
ts = []
for i in gammas:
    tmp = sts.mstats.mquantiles(arr, prob=[1 - (1 - i) / 2])
    ts.append(tmp[0])

intervals = []
for i in ts:
    intervals.append((e_M - np.sqrt(e_D) * i / np.sqrt(N - 1), e_M + np.sqrt(e_D) * i / np.sqrt(N - 1)))

print(intervals)

lens = [i[1] - i[0] for i in intervals]
plt.plot(gammas, lens, label ="Эмпирическая дисперсия")
plt.xlabel("$Значимотсь$")
plt.ylabel("Величина итервалла")
plt.legend(loc='upper left')
plt.show()

t_D = integr.quad(lambda x: (x ** 2 /x), np.exp(1.0), np.exp(2.0))[0] - \
(integr.quad(lambda x: (x * 1.0 / x), np.exp(1.0), np.exp(2.0)))[0]**2

print("t_D=",t_D)

t_intervals = []
for i in ts:
    t_intervals.append((e_M - np.sqrt(t_D) * i / np.sqrt(N - 1), e_M + np.sqrt(t_D) * i / np.sqrt(N - 1)))

print(t_intervals)

t_lens = [i[1] - i[0] for i in t_intervals]
plt.plot(gammas, t_lens, label ="Теоретическая дисперсия")
plt.xlabel("$Значимотсь$")
plt.ylabel("Величина итервалла")
plt.legend(loc='upper left')
plt.show()

plt.plot(gammas, lens, label ="Эмпирическая дисперсия")
plt.plot(gammas, t_lens, label ="Теоретическая дисперсия")
plt.xlabel("$Значимотсь$")
plt.ylabel("Величина итервалла")
plt.legend(loc='upper left')
plt.show()


def inter_N(n):
    Y = random_numbers_sample(n, A, B)
    e_M = sum(Y) / n
    e_D = 0.0
    for i in range(n):
        e_D += (Y[i] - e_M) ** 2
    e_D = e_D / (n - 1)
    t_rv = sts.t(n-1)
    arr = t_rv.rvs(1000000)

    gammas = [0.9, 0.95, 0.98, 0.99]
    ts = []
    for i in gammas:
        tmp = sts.mstats.mquantiles(arr, prob=[1 - (1 - i) / 2])
        ts.append(tmp[0])

    intervals = []
    t_intervals = []
    for i in ts:
        intervals.append((e_M - np.sqrt(e_D) * i / np.sqrt(n - 1), e_M + np.sqrt(e_D) * i / np.sqrt(n - 1)))
        t_intervals.append((e_M - np.sqrt(t_D) * i / np.sqrt(n - 1), e_M + np.sqrt(t_D) * i / np.sqrt(n - 1)))

    lens = [i[1] - i[0] for i in intervals]
    t_lens = [i[1] - i[0] for i in t_intervals]
    lens.sort()
    t_lens.sort()

    #plt.plot(gammas, lens, label="Эмпирическая дисперсия")
    #plt.plot(gammas, t_lens, label="Теоретическая дисперсия")
    #plt.xlabel("Значимотсь (n=" + str(n) + ")")
    #plt.ylabel("Величина итервалла")
    #plt.legend(loc='upper left')
    #plt.show()
    return lens, t_lens

#для gamma = 0.99

#Ns = [30, 50, 70, 100, 150]
Ns = [100]*100
ints = []
t_ints=[]
for i in Ns:
    ints_pair = inter_N(i)
    ints.append(ints_pair[0][-1])
    t_ints.append(ints_pair[1][-1])

dl=sum(ints)/10
t_dl=sum(t_ints)/10
print('Ср.длина доверителного интервалла для эмпирической дисперсии: ', dl)
print('Ср.длина доверителного интервалла для эмпирической дисперсии: ',t_dl)

#plt.plot(Ns, ints)
#plt.xlabel("Размер выборки")
#plt.ylabel("Размер интервалла(эмипирически)")
#plt.show()

#plt.plot(Ns, t_ints)
#plt.xlabel("Размер выборки")
#plt.ylabel("Размер интервалла(теоретически)")
#plt.show()