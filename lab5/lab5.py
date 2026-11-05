import pandas as pd
import numpy as np

file_name = "LAB_RAB_5_DATA.xlsx"
df_v2 = pd.read_excel(file_name, sheet_name="ВАРИАНТ 2")

corr_matrix = df_v2.corr()
print("МАТРИЦА КОРРЕЛЯЦИИ")
print(corr_matrix.round(4))

var_x1 = df_v2['X1'].var()
var_x2 = df_v2['X2'].var()
var_y = df_v2['Y'].var()

print("\n")
print("--VAR--")
print(f"X1: {var_x1:.4f}")
print(f"X2: {var_x2:.4f}")
print(f"Y:  {var_y:.4f}")


cov_matrix = df_v2.cov()
print("\n")
print("МАТРИЦА КОВАРИАЦИИ")
print(cov_matrix.round(4))

cov_x1_y = df_v2['X1'].cov(df_v2['Y'])
cov_x2_y = df_v2['X2'].cov(df_v2['Y'])
cov_x1_x2 = df_v2['X1'].cov(df_v2['X2'])

b1_for_x1 = cov_x1_y / var_x1
b0_for_x1 = df_v2['Y'].mean() - b1_for_x1 * df_v2['X1'].mean()

b1_for_x2 = cov_x2_y / var_x2
b0_for_x2 = df_v2['Y'].mean() - b1_for_x2 * df_v2['X2'].mean()

print("\n")
print(f"Модель Y(X1): b1 = {b1_for_x1:.4f}, b0 = {b0_for_x1:.4f}")
print(f"Модель Y(X2): b1 = {b1_for_x2:.4f}, b0 = {b0_for_x2:.4f}")


y = df_v2['Y']
y_mean = y.mean()

y_hat_x1 = b0_for_x1 + b1_for_x1 * df_v2['X1']
y_hat_x2 = b0_for_x2 + b1_for_x2 * df_v2['X2']

TSS = ((y - y_mean) ** 2).sum()

ESS_x1 = ((y_hat_x1 - y_mean) ** 2).sum()
RSS_x1 = ((y - y_hat_x1) ** 2).sum()

ESS_x2 = ((y_hat_x2 - y_mean) ** 2).sum()
RSS_x2 = ((y - y_hat_x2) ** 2).sum()

print("\n")
print(f"TSS = {TSS:.4f}")
print("\nМодель Y(X1):")
print(f"ESS = {ESS_x1:.4f}")
print(f"RSS = {RSS_x1:.4f}")
print(f"ESS + RSS = {ESS_x1 + RSS_x1:.4f}")
print(f"R^2 = {ESS_x1 / TSS:.4f}")

print("\nМодель Y(X2):")
print(f"ESS = {ESS_x2:.4f}")
print(f"RSS = {RSS_x2:.4f}")
print(f"ESS + RSS = {ESS_x2 + RSS_x2:.4f}")
print(f"R^2 = {ESS_x2 / TSS:.4f}")