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

M = 10 # мощность выброса источника
H = 40 #
U = 4 #
K = 20 #