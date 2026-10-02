import numpy
import pandas
import scipy as sci
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

n = 119
sigma = 1
a = -1

##1 пункт
x = norm.rvs(a, sigma, n)

alfa_counts, alfa_intervals = numpy.histogram(x, bins='scott', density=False)
print("Сумма абсолютных частот = ", numpy.sum(alfa_counts))
print("Сумма относительных частот = ", numpy.sum(alfa_counts/n))

##2.1 пункт
fig, axes = plt.subplots(3, 3, figsize=(18, 9))
for i in range(2, 11):
    axes[(i-2)//3, (i-2)%3].hist(x, bins=i, color='red', density=True)
    axes[(i-2)//3, (i-2)%3].title.set_text(f"k = {i}")
plt.show()

fig, axes = plt.subplots(4, 3, figsize=(18, 9))
for i in range(15, 26):
    axes[(i-15)//3, (i-15)%3].hist(x, bins=i, color='red', density=True)
    axes[(i-15)//3, (i-15)%3].title.set_text(f"k = {i}")
plt.show()

##2.2 пункт
plt.hist(x, bins='scott', color='red', density=False)
plt.title("Гистограмма абсолютных частот по правилу Скотта")
plt.show()

##2.3 пункт
plt.hist(x, bins='scott', color='red', density=True)
pdf = numpy.linspace(min(x), max(x), 1000)
yy = norm.pdf(pdf, a, sigma)
plt.plot(pdf, yy, linewidth=2, label='Теоретическая плотность')
plt.title("Гистограмма относительных частот по правилу Скотта")
plt.show()

##2.4 пункт
plt.hist(x, bins='scott', color='red', density=True, cumulative=True)
cdf = numpy.linspace(min(x), max(x), 1000)
yy1 = norm.cdf(cdf, a, sigma)
plt.plot(cdf, yy1, linewidth=2, label='Теоретическая кумулятивная функция')
plt.title("Гистограмма кумулятивной функции по правилу Скотта")
plt.show()

##2.5 пункт
plt.figure(figsize=(10, 3))
sns.boxplot(x=x)

plt.title("Бокс-плот распределения X")
plt.show()

Q1 = numpy.quantile(x, 0.25)
Q2 = numpy.quantile(x, 0.50)
Q3 = numpy.quantile(x, 0.75)

IQR = Q3 - Q1

lower_whisker = Q1 - 1.5 * IQR
upper_whisker = Q3 + 1.5 * IQR

print("Q1 = ", Q1)
print("Медиана = ", Q2)
print("Q3 = ", Q3)
print("Интерквартильный размах = ", IQR)
print("Нижняя граница усов = ", lower_whisker)
print("Верхняя граница усов = ", upper_whisker)

##3.1 пункт
#способ 1
mean_manual = numpy.sum(x) / n
median_manual = numpy.median(x)

counts, intervals = numpy.histogram(x, bins='scott')
modal_index = numpy.argmax(counts)
mode_manual = (intervals[modal_index] + intervals[modal_index + 1]) / 2

var_manual = numpy.sum((x - mean_manual) ** 2) / n
var_adj_manual = numpy.sum((x - mean_manual) ** 2) / (n - 1)

std_manual = numpy.sqrt(var_manual)
std_adj_manual = numpy.sqrt(var_adj_manual)

skew_manual = (numpy.sum((x - mean_manual) ** 3)) / (n * (std_manual ** 3)) - 3
kurt_manual = (numpy.sum((x - mean_manual) ** 4)) / (n * (std_manual ** 4)) - 3

#способ 2
df = pandas.DataFrame(x, columns=['X'])

mean_auto = numpy.mean(x)
median_auto = numpy.median(x)
mode_auto = float(stats.mode(x, keepdims=False).mode)
var_auto = numpy.var(x, ddof=0)
var_adj_auto = numpy.var(x, ddof=1)
std_auto = numpy.std(x, ddof=0)
std_adj_auto = numpy.std(x, ddof=1)
skew_auto = stats.skew(x)
kurt_auto = stats.kurtosis(x)

results_3_1 = pandas.DataFrame(
    {
        'Параметр':[
            "Среднее",
            "Медиана",
            "Мода",
            "Дисперсия",
            "Исправленная дисперсия",
            "Стандартное отклонение",
            "Исправленное стандартное отклонение",
            "Коэффициент асимметрии",
            "Эксцесс"
        ],
        'Cпособ 1':[
            mean_manual,
            median_manual,
            mode_manual,
            var_manual,
            var_adj_manual,
            std_manual,
            std_adj_manual,
            skew_manual,
            kurt_manual
        ],
        'Cпособ 2':[
            mean_auto,
            median_auto,
            mode_auto,
            var_auto,
            var_adj_auto,
            std_auto,
            std_adj_auto,
            skew_auto,
            kurt_auto
        ]
    }
)

print(results_3_1.to_string(index=False))

## 3.2 пункт
n_large = n * 60
x_large = norm.rvs(a, sigma, n_large)

results_3_2 = pandas.DataFrame(
    {
        "Параметр": [
            "Среднее",
            "Медиана",
            "Дисперсия",
            "Исправленная дисперсия",
            "Стандартное отклонение",
            "Исправленное стандартное отклонение",
            "Коэффициент асимметрии",
            "Эксцесс"
        ],
        "Теоретические значения": [a, a, sigma ** 2, sigma ** 2, sigma, sigma, 0, 0],
        f"Выборка n={n}": [
            mean_auto,
            median_auto,
            var_auto,
            var_adj_auto,
            std_auto,
            std_adj_auto,
            skew_auto,
            kurt_auto
        ],
        f"Выборка n={n_large}": [
            numpy.mean(x_large),
            numpy.median(x_large),
            numpy.var(x_large, ddof=0),
            numpy.var(x_large, ddof=1),
            numpy.std(x_large, ddof=0),
            numpy.std(x_large, ddof=1),
            stats.skew(x_large),
            stats.kurtosis(x_large)
        ],
    }
)

print(results_3_2.to_string(index=False))

## 3.3 пункт

# При увеличении объема выборки в 60 раз (с 119 до 7140) все точечные оценки становятся ближе к своим 
# теоретическим значениям (a = -1, \sigma = 1, асимметрия = 0, эксцесс = 0). Это объясняется законом 
# больших чисел и свойствами состоятельности и несмещенности точечных оценок: с ростом n случайные 
# отклонения усредняются, а погрешность оценивания уменьшается. 