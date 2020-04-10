from math import sqrt, pi, exp
import numpy as np
import matplotlib.pyplot as plt


def tridiagonal_matrix_algorithm(a, b, c, f):
    n = len(f)
    alpha = [0] * n
    beta = [0] * n
    y = [0] * n

    alpha[0] = b[0] / c[0]
    beta[0] = -f[0] / c[0]

    for i in range(1, n):
        alpha[i] = b[i] / (c[i] - alpha[i - 1] * a[i])
        beta[i] = (beta[i - 1] * a[i] - f[i]) / (c[i] - alpha[i - 1] * a[i])

    y[-1] = beta[-1]

    for i in range(n - 2, -1, -1):
        y[i] = alpha[i] * y[i + 1] + beta[i]

    return y


def ca(x):  # аналитическое решение
    return M / (sqrt(U * pi * K * x)) * exp(-H ** 2 * U / (4 * K * x))


M = 10  # Intensity of emission - мощность выброса (в граммах в секунду на метр) источника
H = 40  # высота источника
U = 4  # скорость ветра, направленного вдоль оси x
K = 20  # коэффициент турбулентной диффузии
dz = 1  # шаг по z [м]
dx = .1  # шаг по x [м]

x_cMax = H ** 2 * U / (2 * K)  # точка, в которой концентрация на нулевой высоте достигает максимума

x_b = 100 * x_cMax  # граница по x
z_b = max(4 * H, 100)  # граница по y

n = int(z_b / dz)  # количество отрезков, на которые разбивается исследуемый интервал по z - размерность матрицы

i_s = int(H / dz)  # положение начальной концентрации рассеиваемого вещества (nH должно быть целым числом)
c_i = M / (U * dz)  # начальное значение концентрации
c_0 = [.0] * n
c_0[i_s] = c_i  # вектор начальных концентраций

c = (2 * K + dz ** 2 / dx * U)  # коэффициент на главной диагонали
A = [0.] + [K] * (n - 1)  # диагональ под главной
B = [-1.] + [K] * (n - 1)  # диагональ над главной
C = [-1.] + [c] * (n - 2) + [-1.]  # главная диагональ

c_prev = c_0
f = [0.] * n
p = -dz ** 2 / dx * U  # коэффициент функции

X = [0.0001] + list(np.arange(dx, x_b, dx))
analytic_result = []
numeric_results = []

for i in X:
    analytic_result.append(ca(i))
    numeric_results.append(c_prev)
    for j in range(1, n):
        f[j] = p * c_prev[j]
    c_prev = tridiagonal_matrix_algorithm(A, B, C, f)
transposed_results = list(zip(*numeric_results))

figure, ax = plt.subplots()
ax.plot(X, analytic_result, color='#f12')
ax.plot(X, transposed_results[0], color='#38a')
ax.set(ylabel='c, г/м', xlabel='x, м')
ax.legend(['аналитическое решение', 'численное решение'])
figure.show()

# plt.figure(figsize=[8, 6])
# m = np.array(list(zip(*numeric_results[1:])))
# plt.imshow(m, cmap='hot', aspect='auto')
# plt.gca().invert_yaxis()
# plt.colorbar()
# plt.show()
