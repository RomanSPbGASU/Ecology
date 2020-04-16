import numpy as np
from math import *
from sympy import *

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

k_x= 1 / r
k_y= 1 / r

n = 2  # главный параметр для точности расчёта

N = n ** 2

G_12 = 0.33 * 10 ** 5
G_13 = 0.33 * 10 ** 5
G_23 = 0.33 * 10 ** 5

Jacobi_1 = [] * 5 * N
Deter_1 = [[] * 5 * N] * 5 * N

Jacobi = [] * 5 * N
Deter = [[] * 5 * N] * 5 * N

Hq = [[] * 5 * N] * 5 * N
GG = [] * 5 * N

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
    return cos((2 * i - 1) * i * pi * y / b)


U = 0
V = 0
W = 0
Psi_x = 0
Psi_y = 0

for i in range(n):
    for j in range(n):
        U = U + var(f'u{i}{j}') * X_1(i) * Y_1(j)
        V = W + var(f'v{i}{j}') * X_2(i) * Y_2(j)
        W = W + var(f'w{i}{j}') * X_3(i) * Y_3(j)
        Psi_x = Psi_x + var(f'psi_x{i}{j}') + X_4(i) * Y_4(i)
        Psi_y = Psi_y + var(f'psi_y{i}{j}') + X_5(i) * Y_5(i)

print(U, V, W)

theta_1 = -diff(W, x) / A - k_x * U
theta_2 = -diff(W, y) / B - k_y * V

epsilon_x = diff(U, x) / A + diff(A, y) * V / (A * B) - k_x * W + 1 / 2 * theta_1  **  2
epsilon_y = diff(V, y) / B + diff(B, x) * U / (A * B) - k_y * W + 1 / 2 * theta_2 ** 2

gammax_y = diff(V, x) / A + diff(U, y) / B - diff(A, y) * U / (A * B) - diff(B, x) * V / (A * B) + theta_1 * theta_2

gamma_xz = k * distributed_stress(z)(Psi_x - theta_1)
gamma_yz = k * distributed_stress(z)(Psi_y - theta_2)

varkappa_1 = diff(Psix, x)/A + diff(A, y)*Psiy/(A*B)
varkappa_2 = diff(Psiy, y)/B + diff(B, x)*Psix/(A*B)

varkappa_12 = 1/2*(diff(Psiy, x)/A + diff(Psix, y)/B - (diff(A, y)*Psi_x+ diff(B, x)*Psiy)/(A*B))


M_x= E_1*h ** 3/12*(mu_21*varkappa_2 + varkappa_1)/(-mu_12*mu_21 + 1)
M_y= E_2*h ** 3/12*(mu_12*varkappa_1 + varkappa_2)/(-mu_12*mu_21 + 1)
Mx_y= G_12*h ** 3/6*varkappa_12
My_x= G_12*h ** 3/6*varkappa_12
N_x= E_1*h*(mu_21*epsilony` + `&varepsilonx)/(-mu_12*mu_21 + 1)
N_y= E_2*h*(mu_12*epsilonx` + `&varepsilony)/(-mu_12*mu_21 + 1)

Nx_y= G_12*h*gammax_yNy_x= G_12*h*gammax_yP_x= 0
P_y= 0
Q_x= G_13*k*h*(Psi_x- theta_1)
Q_y= G_23*k*h*(Psi_y- theta_2)

Q_ytime() - st
Ep = 1/2*ApproximateInt(ApproximateInt((Nx*epsilonx` + Ny*`&varepsilony` + 1/2*(Nx_y+ Nyx)*`&gammax_y+ Mx*varkappa_1 + My*varkappa_2 + (Mx_y+ Myx)*varkappa_12 + Qx*(Psi_x- theta_1) + Qy*(Psi_y- theta_2))*A*B, y = 0 .. b, method = simpson), x = a_1 .. a, method = simpson)


time() - st
AA = ApproximateInt(ApproximateInt((Px*U + Py*V + W*q)*A*B, y = 0 .. b, method = simpson), x = a_1 .. a, method = simpson)
Ep = Ep
AA = AA
time() - st
Es = Ep - AA
time() - st

k = 0
for i to n do
    for j to n do k = k + 1 Jacobi[k] = diff(Es, cat(u, i, j)) Jacobi[k + N] = diff(Es, cat(v, i, j)) Jacobi[k + 2*N] = diff(Es, cat(w, i, j)) Jacobi[k + 3*N] = diff(Es, cat(psix`, i, j)) Jacobi[k + 4*N] = diff(Es, cat(`&psiy, i, j)) end do
end do
Jacobi[1]
for l to 5*N do
    k = 0
    for i to n do
        for j to n do k = k + 1 Deter[l, k] = diff(Jacobi[l], cat(u, i, j)) Deter[l, k + N] = diff(Jacobi[l], cat(v, i, j)) Deter[l, k + 2*N] = diff(Jacobi[l], cat(w, i, j)) Deter[l, k + 3*N] = diff(Jacobi[l], cat(psix`, i, j)) Deter[l, k + 4*N] = diff(Jacobi[l], cat(`&psiy, i, j)) end do
    end do
end do


Prob_3 = matrix(5*N, 5*N, [])

MAX = 320
epsillon = 10 ** (-5)
delq = 0.01
qq = 0
BufV = vector(5*N, [])
Buf = vector(5*N, [])
Coef = vector(5*N, [])
for l to 5*N do
    Coef[l] = 0
end do


AnsMatr = matrix(MAX, 1 + 5*N, [])
for p to MAX do
    del = 1
    for m to 100 while epsillon < del do
        for l to 5*N do BufV[l] = Coef[l] end do
        k = 0
        for i to n do
            for j to n do k = k + 1 cat(u, i, j) = Coef[k] cat(v, i, j) = Coef[k + N] cat(w, i, j) = Coef[k + 2*N] cat(psix`, i, j) = Coef[k + 3*N] cat(`&psiy, i, j) = Coef[k + 4*N] end do
        end do
        for i to 5*N do
            for j to 5*N do Jacobi_1[i] = subs({q = qq}, Jacobi[i]) Deter_1[i, j] = subs({q = qq}, Deter[i, j]) end do
        end do
        for l to 5*N do
            Buf[l] = Coef[l]
        end do
        Rans = multiply(inverse(Deter_1), Jacobi_1)
        for l to 5*N do
            Coef[l] = evalf(Buf[l] - Rans[l])
        end do
        del = abs(evalf(BufV[1] - Coef[1]))
        for l to 5*N do
            if del < abs(evalf(BufV[l] - Coef[l])) then del = abs(evalf(BufV[l] - Coef[l])) end if
        end do
    end do
    for l to 5*N do
        AnsMatr[p, l + 1] = Coef[l]
    end do
    AnsMatr[p, 1] = qq
    AnsMatr[p, 2] = subs({x = a/2, y = b/2}, W)
    AnsMatr[p, 3] = subs({x = a/4, y = b/4}, W)
    qq = qq + delq
end do
evalm(AnsMatr)
with(plots)

gr_3 = pointplot([seq([AnsMatr[i, 2], AnsMatr[i, 1]], i = 1 .. MAX)], color = ffc600, symbol = soliddiamond, symbolsize = 15, axis = [gridlines = [10, color = black]], labels = ["W, м", "q, МПа"], legend = "W(a/2, b/2)")
gr_4 = pointplot([seq([AnsMatr[i, 3], AnsMatr[i, 1]], i = 1 .. MAX)], color = "Teal", symbol = soliddiamond, symbolsize = 15, axis = [gridlines = [10, color = black]], labels = ["W, м", "q, МПа"], legend = "W(a/4, b/4)")

print(display([gr_3, gr_4]))

gr_3 = pointplot([seq([AnsMatr[i, 2], AnsMatr[i, 1]], i = 290 .. 300)], color = ffc600, symbol = soliddiamond, symbolsize = 25, axis = [gridlines = [10, color = black]], labels = ["W", "q"], legend = "W(a/2, b/2)")
gr_4 = pointplot([seq([AnsMatr[i, 3], AnsMatr[i, 1]], i = 290 .. 300)], color = "Teal", symbol = soliddiamond, symbolsize = 25, axis = [gridlines = [10, color = black]], labels = ["W", "q"], legend = "W(a/4, b/4)")

print(display([gr_3, gr_4]))
print(display(plot3d()))
