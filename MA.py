# corr_matrix = [
#     [1.000, -0.379, -0.366, -0.060, -0.552, -0.663],
#     [-0.379, 1.000, -0.243, -0.126, 0.067, 0.140],
#     [-0.366, -0.243, 1.000, -0.358, 0.208, 0.254],
#     [-0.060, -0.126, -0.358, 1.000, -0.183, -0.068],
#     [-0.552, 0.067, 0.208, -0.183, 1.000, 0.618],
#     [-0.663, 0.140, 0.254, -0.068, 0.618, 1.000]
# ]
#
# print(corr_matrix)
# for i in range(0, 6):
#     print(corr_matrix[i][i])
#
# M = 10
# H = 100
# U = 10
# K = 50
#
# x_max = H^2*U/(2*K)
#
# pr_gr = 4*x_max
#
# h_x = 250
#
# h_z = H/4


# import pandas as pd
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# from scipy import stats
# # matplotlib.style.use('ggplot')
# # %matplotlib inline
#
# df = pd.read_csv('ArishaStatistics.csv', sep=';', decimal=',')
# df.head()
#
# df.describe()
# df.pop(1)
import math
import numpy as np
import matplotlib.pyplot as plt

M = 10  # мощность выброса (в граммах в секунду на метр) источника
H = 40  # высота источника
U = 4  # скорость ветра, направленного вдоль оси x
K = 20  # коэффициент турбулентной диффузии

x_max = H ** 2 * U / (2 * K)  # точка, в которой концентрация на нулевой высоте достигает максимума


# print(x_max)


def ca(x):  # аналитическое решение
    return M / (math.sqrt(U * math.pi * K * x)) * math.exp(-H ** 2 * U / (4 * K * x))


pr_gr = int(4 * x_max)
# print(pr_gr, type(pr_gr))

dots = []

for i in range(1, pr_gr):
    dots.append(ca(i))


h_x = 250
h_z = 25

CC = (2 * K + h_z ** 2 / h_x * U)

hr = H / h_z

el2 = M / (U * h_z)

d_size = 16
MA = np.eye(d_size, k=-1) * K + np.eye(d_size) * CC + np.eye(d_size, k=1) * K
MA[0][0] = -1
MA[0][1] = -1
MA[d_size - 1][d_size - 1] = -1
MA[d_size - 1][d_size - 2] = 0
# print(MA)

F = np.zeros(shape=[16], dtype=int)
F[5] = M / U * h_z
print(F)

for i, elem in enumerate(F):
    F[i] = -h_z ** 2 / h_x * U * elem

A = np.diag(MA, k=-1)
B = np.diag(MA, k=1)
C = np.diag(MA)


# print(A, '\n', B, '\n', C)


def tridiagonal_matrix_algorithm(a, b, c, f):
    alpha = [0]
    beta = [0]
    n = len(f)
    x = [0] * n


#    a, b, c, f = map(lambda k_list: map(float, k_list), (a, b, c, f))

    for i in range(n - 1):
        alpha.append(-b[i] / (a[i] * alpha[i] + c[i]))
        beta.append((f[i] - a[i] * beta[i]) / (a[i] * alpha[i] + c[i]))

    x[n - 1] = (f[n - 1] - a[n - 2] * beta[n - 1]) / (c[n - 1] + a[n - 2] * alpha[n - 1])

    for i in reversed(range(n - 1)):
        x[i] = alpha[i + 1] * x[i + 1] + beta[i + 1]

    return x


numeric = tridiagonal_matrix_algorithm(A, B, C, list(F))
print(numeric)

#plt.plot(dots, color='#f12')
plt.plot(numeric, color='#b48')
plt.ylabel('c, г/м')
plt.xlabel('x, м')
plt.legend(['аналитическое решение', 'численное решение'])
plt.show()
