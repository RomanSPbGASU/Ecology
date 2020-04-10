from math import sqrt, pi, exp
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


M = 10  # мощность выброса (в граммах в секунду на метр) источника
H = 40  # высота источника
U = 4  # скорость ветра, направленного вдоль оси x
K = 20  # коэффициент турбулентной диффузии
dz = 1  # шаг по z [м]
dx = 1  # шаг по x [м]

x_max = H ** 2 * U / (2 * K)  # точка, в которой концентрация на нулевой высоте достигает максимума

x_m = int(4 * x_max)  # граница по x
z_n = max(4 * H, 100)  # граница по y
# print(x_m, ' ', z_n)

# TODO: исправить приведение к int - выразить целочисленно
n = int(z_n / dz)  # количество отрезков, на которые разбивается исследуемый интервал по z - размерность матрицы
m = int(x_m / dx)  # количество отрезков по x - количество "слоёв" для расчёта
i_s = int(H / dz)  # положение начальной концентрации рассеиваемого вещества (nH должно быть целым числом)
c_i = M / (U * dz)  # начальное значение концентрации
c_0 = [.0] * n
c_0[i_s] = c_i  # вектор начальных концентраций
# print("c_o", c_0)

c = (2 * K + dz ** 2 / dx * U)  # коэффициент на главной диагонали
A = [0] + [K] * (n - 1)  # диагональ под главной
B = [-1] + [K] * (n - 1)  # диагональ над главной
C = [-1] + [c] * (n - 2) + [-1]  # главная диагональ
# print(A, '\n', B, '\n', C)

c_prev = c_0
results = [[0.] * n] * (m + 1)
f = [0] * n
p = -dz ** 2 / dx * U  # коэффициент функции

results[0] = c_prev

for xi in range(1, m + 1):
    for i in range(1, n):
        f[i] = p * c_prev[i]

    c = tridiagonal_matrix_algorithm(A, B, C, f)
    c_prev = c.copy()
    results[xi] = c

print(results)


def ca(x):  # аналитическое решение
    return M / (sqrt(U * pi * K * x)) * exp(-H ** 2 * U / (4 * K * x))


dots = []

for i in range(1, x_m):
    dots.append(ca(i))

plt.plot(dots, color='#f12')
transposed_results = list(zip(*results))
plt.plot(transposed_results[0], color='#38a')
plt.ylabel('c, г/м')
plt.xlabel('x, м')
plt.legend(['аналитическое решение', 'численное решение'])
plt.show()
