import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

a = -1.0  
sigma_sq = 1.0  
sigma = math.sqrt(sigma_sq)
n = 9 
N = 600  
gamma = 0.915 

##1 пункт
X_samples = stats.norm.rvs(loc=a, scale=sigma, size=(N, n))  # матрица N выборок по n значений из N(a, sigma^2)
X_bars = np.mean(X_samples, axis=1)  # выборочное среднее каждой выборки (по строкам)


##2 пункт
plt.figure(figsize=(10, 5))

plt.hist(
    X_bars,
    bins="scott", 
    density=True, 
    color="skyblue",
    edgecolor="black",
    alpha=0.7,
    label="Гистограмма относительных частот $\\overline{X}$",
)

x_grid = np.linspace(min(X_bars), max(X_bars), 500)  # сетка точек для гладкой кривой
y_pdf = stats.norm.pdf(x_grid, loc=a, scale=sigma / math.sqrt(n))  # теор. плотность: X_bar ~ N(a, sigma^2/n)

plt.plot(
    x_grid,
    y_pdf,
    "r-",
    linewidth=2,
    label=f"Теоретическая плотность $N({a}, {sigma_sq/n:.4f})$",
)

plt.title("Гистограмма относительных частот и теоретическая плотность $\\overline{X}$")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

## 3 пункт
plt.figure(figsize=(8, 3))
sns.boxplot(x=X_bars, color="skyblue")

plt.title("Бокс-плот распределения выборочных средних $\\overline{X}$")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

## 4 пункт
mean_Xbar = np.mean(X_bars)
var_Xbar = np.var(X_bars, ddof=0) 
std_Xbar = np.std(X_bars, ddof=0)
median_Xbar = np.median(X_bars)
skew_Xbar = stats.skew(X_bars)
kurt_Xbar = stats.kurtosis(X_bars)

df_estimates = pd.DataFrame(
    {
        "Параметр": [
            "Математическое ожидание (выборочное среднее)",
            "Дисперсия",
            "Стандартное отклонение",
            "Медиана",
            "Коэффициент асимметрии",
            "Эксцесс",
        ],
        "Точечная оценка для X_bar": [
            mean_Xbar,
            var_Xbar,
            std_Xbar,
            median_Xbar,
            skew_Xbar,
            kurt_Xbar,
        ],
    }
)

print(df_estimates.to_string(index=False))

## 5 пункт
pct_gt_a = np.mean(X_bars > a) * 100  # среднее булева массива = доля True;

print('\n')
print(f"Параметр a = {a}")
print(f"Процент значений X_bar > a: {pct_gt_a:.2f}%")

## 6 пункт
D_B = np.var(X_samples, axis=1, ddof=0)  # выборочная дисперсия D_B для каждой из N выборок

## 7 пункт
pct_lt_sigma2 = np.mean(D_B < sigma_sq) * 100  # доля выборок, где D_B < истинной sigma^2

print('\n')
print(f"Параметр sigma^2 = {sigma_sq}")
print(f"Процент значений D_B < sigma^2: {pct_lt_sigma2:.2f}%")

## 8 пункт
Y = n * D_B / sigma_sq  # статистика Y = n*D_B/sigma^2 ~ chi^2(n-1)

## 9 пункт
plt.figure(figsize=(10, 5))

plt.hist(
    Y,
    bins="scott",
    density=True,
    color="orange",
    edgecolor="black",
    alpha=0.7,
    label="Гистограмма относительных частот $Y$",
)

df_chi2 = n - 1
y_grid = np.linspace(min(Y), max(Y), 500)
plt.plot(
    y_grid,
    stats.chi2.pdf(y_grid, df=df_chi2),
    "r-",
    linewidth=2,
    label=f"Теоретическая плотность $\\chi^2({df_chi2})$",
)

plt.title("Гистограмма и теоретическая плотность $\\chi^2$ случайной величины $Y$")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

## 10 пункт
plt.figure(figsize=(8, 3))
sns.boxplot(x=Y, color="orange")

plt.title("Бокс-плот случайной величины $Y$")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

## 11 пункт
k = n - 1

E_Y_th = stats.chi2.mean(df=k)
Var_Y_th = stats.chi2.var(df=k)
Med_Y_th = stats.chi2.median(df=k)
Skew_Y_th = float(stats.chi2.stats(df=k, moments="s"))
Kurt_Y_th = float(stats.chi2.stats(df=k, moments="k"))

print('\n')
print(f"Математическое ожидание M[Y]: {E_Y_th:.4f}")
print(f"Дисперсия D[Y]: {Var_Y_th:.4f}")
print(f"Медиана Me[Y]: {Med_Y_th:.4f}")
print(f"Коэффициент асимметрии: {Skew_Y_th:.4f}")
print(f"Эксцесс: {Kurt_Y_th:.4f}")

## 12 пункт
E_Y_emp = np.mean(Y)
Med_Y_emp = np.median(Y)
Var_Y_emp = np.var(Y, ddof=0)
Skew_Y_emp = stats.skew(Y)
Kurt_Y_emp = stats.kurtosis(Y)

df_Y_comp = pd.DataFrame(
    {
        "Характеристика": [
            "Математическое ожидание",
            "Медиана",
            "Дисперсия",
            "Коэффициент асимметрии",
            "Эксцесс",
        ],
        "Теоретическое значение (п. 11)": [
            E_Y_th,
            Med_Y_th,
            Var_Y_th,
            Skew_Y_th,
            Kurt_Y_th,
        ],
        "Выборочная оценка (п. 12)": [
            E_Y_emp,
            Med_Y_emp,
            Var_Y_emp,
            Skew_Y_emp,
            Kurt_Y_emp,
        ],
    }
)

print('\n')
print(df_Y_comp.to_string(index=False))

## 13 пункт
x_single = stats.norm.rvs(loc=a, scale=sigma, size=n)  # одна выборка объёма n
mean_s = np.mean(x_single)  # её выборочное среднее

# Способ 1: Программная реализация формул с использованием ppf
# Формула: x_bar ± z_crit * (sigma / sqrt(n))
alpha = 1 - gamma  # уровень значимости
z_crit = stats.norm.ppf(1 - alpha / 2)  # квантиль N(0,1) уровня 1 - alpha/2 (двусторонний интервал)
margin = z_crit * (sigma / math.sqrt(n))  # полуширина интервала (sigma известна -> z-интервал)
ci_manual = (mean_s - margin, mean_s + margin)  # границы доверительного интервала для a

# Способ 2: Использование встроенного метода stats.norm.interval
ci_auto = stats.norm.interval(confidence=gamma, loc=mean_s, scale=sigma / math.sqrt(n))  # центр — x_bar, масштаб — стандартная ошибка среднего

print('\n')
print(f"Выборочное среднее (x_bar): {mean_s:.4f}")
print(f"Способ 1 (по формулам с ppf): ({ci_manual[0]:.4f}, {ci_manual[1]:.4f})")
print(f"Способ 2 (stats.norm.interval): ({ci_auto[0]:.4f}, {ci_auto[1]:.4f})")