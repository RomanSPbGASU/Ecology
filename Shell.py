import numpy as np
from sympy import symbols, diff, integrate, sin, cos, pi
from sympy.integrals.quadrature import gauss_legendre
from matplotlib import pyplot as plt

E_1 = 2.1 * 10 ** 5
E_2 = 2.1 * 10 ** 5
mu_12 = 0.3
mu_21 = 0.3
h = 0.09
z = -h / 2
r = 225 * h
k = 5 / 6


def distributed_stress(i):
    return 6 * (1 / 4 - i ** 2 / h ** 2)


a = 60 * h
a_1 = 0
b = 60 * h

A = 1
B = 1

k_x = 1 / r
k_y = 1 / r

n = 1  # главный параметр для точности расчёта

N = n ** 2

G_12 = 0.33 * 10 ** 5
G_13 = 0.33 * 10 ** 5
G_23 = 0.33 * 10 ** 5

Jacobi_1 = [None] * 5 * N
Deter_1 = [[None] * 5 * N] * 5 * N

Jacobi = [None] * 5 * N
Deter = [[None] * 5 * N] * 5 * N

Hq = [[None] * 5 * N] * 5 * N
GG = [None] * 5 * N

x, y = symbols('x y')


def X_1(i):
    return sin(2 * i * pi * x / a)


def X_2(i):
    return sin((2 * i - 1) * pi * x / a)


def X_3(i):
    return sin((2 * i - 1) * pi * x / a)


def X_4(i):
    return cos((2 * i - 1) * pi * x / a)


def X_5(i):
    return sin((2 * i - 1) * pi * x / a)


def Y_1(i):
    return sin((2 * i - 1) * i * pi * y / b)


def Y_2(i):
    return sin(2 * i * pi * y / b)


def Y_3(i):
    return sin((2 * i - 1) * pi * y / b)


def Y_4(i):
    return sin((2 * i - 1) * pi * y / b)


def Y_5(i):
    return cos((2 * i - 1) * pi * y / b)


U = 0
V = 0
W = 0
Psi_x = 0
Psi_y = 0

u = symbols(f'u:{n}:{n}')
v = symbols(f'u:{n}:{n}')
w = symbols(f'u:{n}:{n}')
psi_x = symbols(f'u:{n}:{n}')
psi_y = symbols(f'u:{n}:{n}')

for i in range(n):
    for j in range(n):
        U = U + u[i*j] * X_1(i) * Y_1(j)
        V = V + v[i*j] * X_2(i) * Y_2(j)
        W = W + w[i*j] * X_3(i) * Y_3(j)
        Psi_x = Psi_x + psi_x[i*j] * X_4(i) * Y_4(i)
        Psi_y = Psi_y + psi_y[i*j] * X_5(i) * Y_5(i)

print(U, V, W)

theta_1 = -diff(W, x) / A - k_x * U
theta_2 = -diff(W, y) / B - k_y * V

epsilon_x = diff(U, x) / A + diff(A, y) * V / (A * B) - k_x * W + 1 / 2 * theta_1 ** 2
epsilon_y = diff(V, y) / B + diff(B, x) * U / (A * B) - k_y * W + 1 / 2 * theta_2 ** 2

gammax_y = diff(V, x) / A + diff(U, y) / B - diff(A, y) * U / (A * B) - diff(B, x) * V / (A * B) + theta_1 * theta_2

gamma_xz = k * distributed_stress(z) * (Psi_x - theta_1)
gamma_yz = k * distributed_stress(z) * (Psi_y - theta_2)

kappa_1 = diff(Psi_x, x) / A + diff(A, y) * Psi_y / (A * B)
kappa_2 = diff(Psi_y, y) / B + diff(B, x) * Psi_x / (A * B)

kappa_12 = 1 / 2 * (diff(Psi_y, x) / A + diff(Psi_x, y) / B - (diff(A, y) * Psi_x + diff(B, x) * Psi_y) / (A * B))

M_x = E_1 * h ** 3 / 12 * (mu_21 * kappa_2 + kappa_1) / (-mu_12 * mu_21 + 1)
M_y = E_2 * h ** 3 / 12 * (mu_12 * kappa_1 + kappa_2) / (-mu_12 * mu_21 + 1)
M_xy = G_12 * h ** 3 / 6 * kappa_12
M_yx = G_12 * h ** 3 / 6 * kappa_12
N_x = E_1 * h * (mu_21 * epsilon_y + epsilon_x) / (-mu_12 * mu_21 + 1)
N_y = E_2 * h * (mu_12 * epsilon_x + epsilon_y) / (-mu_12 * mu_21 + 1)
N_xy = G_12 * h * gammax_y
N_yx = G_12 * h * gammax_y

P_x = 0
P_y = 0
Q_x = G_13 * k * h * (Psi_x - theta_1)
Q_y = G_23 * k * h * (Psi_y - theta_2)

Ep = 1 / 2 * integrate(integrate(
    (N_x * epsilon_x + N_y * epsilon_y + 1 / 2 * (N_xy + N_yx) * gammax_y + M_x * kappa_1 + M_y * kappa_2 + (
            M_xy + M_yx) * kappa_12 + Q_x * (Psi_x - theta_1) + Q_y * (Psi_y - theta_2)) * A * B, (y, 0, b)),
    (x, a_1, a))

q = symbols('q')
AA = integrate(integrate((P_x * U + P_y * V + W * q) * A * B, (y, 0, b)), (x, a_1, a))
Ep = Ep
AA = AA

Es = Ep - AA

k = 0
for i in range(n):
    for j in range(n):
        k += 1
        Jacobi[k] = diff(Es, u[i*j])
        Jacobi[k + N] = diff(Es, v[i*j])
        Jacobi[k + 2 * N] = diff(Es, w[i*j])
        Jacobi[k + 3 * N] = diff(Es, psi_x[i*j])
        Jacobi[k + 4 * N] = diff(Es, psi_y[i*j])

for l in range(5 * N):
    k = 0
    for i in range(n):
        for j in range(n):
            k = k + 1
            Deter[l][k] = diff(Jacobi[l], u[i*j])
            Deter[l][k + N] = diff(Jacobi[l], v[i*j])
            Deter[l][k + 2 * N] = diff(Jacobi[l], w[i*j])
            Deter[l][k + 3 * N] = diff(Jacobi[l], psi_x[i*j])
            Deter[l][k + 4 * N] = diff(Jacobi[l], psi_y[i*j])

Prob_3 = [[] * 5 * N] * 5 * N

MAX = 320
epsillon = 10 ** (-5)
delq = 0.01
qq = 0
BufV = [] * 5 * N
Buf = [] * 5 * N
Coef = [] * 5 * N

for l in range(5 * N):
    Coef[l] = 0

AnsMatr = [[] * MAX] * 5 * N

for p in range(MAX):
    delta = 0
    for m in range(100):
        if delta <= epsillon:
            break

        for l in range(5 * N):
            Buf[l] = Coef[l]

        k = 0

        for i in range(n):
            for j in range(n):
                k = k + 1
                u[i*j] = Coef[k]
                v[i*j] = Coef[k + N]
                w[i*j] = Coef[k + 2 * N]
                psi_x[i*j] = Coef[k + 3 * N]
                psi_y[i*j] = Coef[k + 4 * N]

        for i in range(5 * N):
            for j in range(5 * N):
                Jacobi_1[i] = Jacobi[i].subs(q, qq)
                Deter_1[i*j] = Deter[i*j].subs(q, qq)

            for l in range(5 * N):
                Buf[l] = Coef[l]

            Rans = Deter_1 ** -1 * Jacobi_1

            for l in range(5 * N):
                Coef[l] = (Buf[l] - Rans[l]).evalf()

            delta = abs((BufV[1] - Coef[1]).evalf())

            for l in range(5 * N):
                if abs((BufV[l] - Coef[l]).evalf()) > delta:
                    delta = abs((BufV[l] - Coef[l]).evalf())

        for l in range(5 * N):
            AnsMatr[p][l + 1] = Coef[l]

        AnsMatr[p][1] = qq
        AnsMatr[p][2] = W.subs([(x, a / 2), (y, b / 2)])
        AnsMatr[p][3] = W.subs([(x, a / 4), (y, b / 4)])
        qq = qq + delq

    AnsMatr.evalm()

Matrix = AnsMatr

# A = list(zip(*Matrix2))[:2]
# A = A[1], A[0]
# B = list(zip(*Matrix2))[0:3:2]
# B = B[1], B[0]

C = list(zip(*Matrix))[:2]
C = C[1], C[0]
D = list(zip(*Matrix))[0:3:2]
D = D[1], D[0]

figure, axes = plt.subplots(2, 2)

# axes[0, 0].plot(*A, '.,', color='#f12', linewidth=11)
# axes[0, 0].plot(*B, '.,', color='#38a', linewidth=11)
# axes[0, 0].set(ylabel='q, МПа', xlabel='W, м')
# axes[0, 0].legend(['W(a/2, b/2)', 'W(a/4, b/4)'])
# axes[0, 0].grid(color='black', linestyle=':', linewidth=0.2)

axes[0, 1].plot(*C, color='#f12')
axes[0, 1].plot(*D, color='#38a')
axes[0, 1].set(ylabel='q, МПа', xlabel='W, м')
axes[0, 1].legend(['W(a/2, b/2)', 'W(a/4, b/4)'])
axes[0, 1].grid(color='black', linestyle=':', linewidth=0.2)
#
# axes[1, 0].plot(*A, color='#f12')
# axes[1, 0].plot(*C, color='#38a')
# axes[1, 0].set(ylabel='q, МПа', xlabel='W, м')
# axes[1, 0].legend(['W(a/2, b/2)', 'W(a/4, b/4)'])
# axes[1, 0].grid(color='black', linestyle=':', linewidth=0.2)
#
# axes[1, 1].plot(*B, color='#f12')
# axes[1, 1].plot(*D, color='#38a')
# axes[1, 1].set(ylabel='q, МПа', xlabel='W, м')
# axes[1, 1].legend(['W(a/2, b/2)', 'W(a/4, b/4)'])
# axes[1, 1].grid(color='black', linestyle=':', linewidth=0.2)

figure.show()
