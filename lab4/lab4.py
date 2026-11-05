import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import openpyxl

warnings.filterwarnings("ignore")

# =====================================================================
# ВАРИАНТ 2
# Исследуемый ФО: ЦФО | Округ-сосед: СЗФО | Пара лет (п.5): 2010 и 2021
# Уровень значимости: 0.01
# =====================================================================
FO_NAME = "ЦФО"
NB_NAME = "СЗФО"
YEARS = [2005, 2010, 2015, 2019, 2020, 2021]
YEAR_PAIR = (2010, 2021)
ALPHA = 0.01
LABEL_X = "Число врачей на 10 000 чел. населения"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XLSX_PATH = os.path.join(BASE_DIR, "CHISLO_DOCTORS.xlsx")
YEAR_COL = {y: 7 + i for i, y in enumerate(YEARS)}

wb = openpyxl.load_workbook(XLSX_PATH, data_only=True)
rows = list(wb["ROSSTAT"].iter_rows(min_row=1, max_row=104, values_only=True))

RF_ROW = 5

FO_ROWS = {
    "ЦФО": list(range(7, 25)),
    "СЗФО": [26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 38],
    }


def get_sample(fo, year):
   """Извлекает массив значений показателя X по регионам округа fo за выбранный year."""
   return np.array([rows[r - 1][YEAR_COL[year]] for r in FO_ROWS[fo]], dtype=float)


def get_rf(year):
    """Возвращает общероссийское значение показателя X (значение по РФ) за выбранный year."""
    return float(rows[RF_ROW - 1][YEAR_COL[year]])


def decision(p):
    """
    Принимает статистическое решение по p-value на основе уровня значимости ALPHA:
    если p-value < ALPHA, то нулевая гипотеза H0 отвергается.
    """
    return "отвергаем H0" if p < ALPHA else "нет оснований отвергнуть H0"


# =====================================================================
# П.1-2. Выборки, описательная статистика, графики
# =====================================================================
print("=" * 72)
print(f"П.1-2. Описательная статистика X ({FO_NAME}), n = {len(FO_ROWS[FO_NAME])}")
print("=" * 72)

desc = []
for year in YEARS:
    s = get_sample(FO_NAME, year)  # Выборка по ЦФО за текущий год
    desc.append({
        "Год": year,
        "Среднее": s.mean(),
        "СКО": s.std(ddof=1),
        "Медиана": np.median(s),
        "Асимметрия": stats.skew(s),
        "Эксцесс": stats.kurtosis(s),
    })

# Выводим таблицу описательной статистики
print(pd.DataFrame(desc).round(4).to_string(index=False))

# Построение гистограмм с наложенной нормальной кривой
fig, axes = plt.subplots(
    2, 3, figsize=(15, 8)
)
for ax, year in zip(axes.ravel(), YEARS):
    s = get_sample(FO_NAME, year)
    # Гистограмма плотности эмпирического распределения
    sns.histplot(
        s,
        bins=6,
        stat="density",
        color="skyblue",
        edgecolor="black",
        label="Гистограмма",
        ax=ax,
    )
    # Сетка точек по оси X для построения гладкой теоретической кривой
    grid = np.linspace(s.min() - s.std(), s.max() + s.std(), 200)
    # Плотность нормального распределения с параметрами выборки (среднее и СКО)
    ax.plot(grid, stats.norm.pdf(grid, s.mean(), s.std(ddof=1)), "r--", label="Нормальная кривая",)
    ax.set_title(f"{FO_NAME}, {year} г.")
    ax.set_xlabel(LABEL_X)
    ax.set_ylabel("Плотность")
    ax.legend()

fig.suptitle(f"Гистограммы показателя X, {FO_NAME}")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "hist_cfo.png"), dpi=150)
plt.show()

# Построение боксплотов
fig, axes = plt.subplots(
    2, 3, figsize=(15, 8)
)
for ax, year in zip(axes.ravel(), YEARS):
    s = get_sample(FO_NAME, year)
    # Боксплот показывает медиану, квартили и выбросы
    sns.boxplot(y=s, color="skyblue", ax=ax)
    # Красная пунктирная линия показывает среднее значение выборки
    ax.axhline(s.mean(), color="red", linestyle="--", label="Среднее")
    ax.set_title(f"{FO_NAME}, {year} г.")
    ax.set_xlabel(f"{year} г.")
    ax.set_ylabel(LABEL_X)
    ax.legend()

fig.suptitle(f"Диаграммы размаха показателя X, {FO_NAME}")
plt.tight_layout()
plt.savefig(
    os.path.join(BASE_DIR, "box_cfo.png"), dpi=150
)
plt.show()

# ===========================================================================
# П.3. Проверка нормальности ЦФО (Шапиро-Уилк), H0: распределение нормальное
# ===========================================================================
print("\n" + "=" * 72)
print(f"П.3. Критерий Шапиро-Уилка, {FO_NAME}, альфа = {ALPHA}")
print("=" * 72)

normal_fo = ({})
sw_rows = []
for year in YEARS:
    # Оценка нормальности распределения с помощью критерия Шапиро–Уилка
    w, p = stats.shapiro(get_sample(FO_NAME, year))
    normal_fo[year] = p > ALPHA  # Если p > alpha, принимаем нормальность
    sw_rows.append({
        "Год": year,
        "W": round(w, 4),  # Значение статистики критерия W
        "p-value": round(p, 4),  # Достигнутый уровень значимости (p-value)
        "Вывод": "нормальное" if p > ALPHA else "отлично от нормального",
    })

print(pd.DataFrame(sw_rows).to_string(index=False))

# =====================================================================
# П.4. ЦФО vs СЗФО (независимые выборки)
# =====================================================================
print("\n" + "=" * 72)
print(f"П.4. Сравнение {FO_NAME} и {NB_NAME} (независимые выборки)")
print("=" * 72)

for year in YEARS:
    x = get_sample(FO_NAME, year)  # Выборка по ЦФО
    y = get_sample(NB_NAME, year)  # Выборка по СЗФО

    # Проверяем нормальность распределения выборки округа-соседа (СЗФО)
    p_nb = stats.shapiro(y)[1]
    normal_nb = p_nb > ALPHA
    # Переменная показывает, нормальны ли ОБЕ выборки одновременно
    both_normal = normal_fo[year] and normal_nb

    # Формируем альтернативную гипотезу H1 (односторонняя проверка):
    # Если распределения нормальны — сравниваем средние выборок, иначе — медианы
    if both_normal:
        alt = "greater" if x.mean() > y.mean() else "less"
    else:
        alt = "greater" if np.median(x) > np.median(y) else "less"
    sign = ">" if alt == "greater" else "<"

    print(f"\n--- {year} г. ---")
    print(f"H0: M({FO_NAME}) = M({NB_NAME});  H1: M({FO_NAME}) {sign} M({NB_NAME})")
    print(f"Шапиро-Уилк: {FO_NAME} - {'нормальное' if normal_fo[year] else 'не нормальное'}, "
        f"{NB_NAME} - {'нормальное' if normal_nb else 'не нормальное'} (p = {p_nb:.4f})"
    )

    # Выбор подходящего статистического критерия:
    if both_normal:
        # Для нормальных выборок проверяем равенство дисперсий (тест Левена)
        p_lev = stats.levene(x, y)[1]
        equal_var = p_lev > ALPHA
        print(
            f"Левен: p = {p_lev:.4f} -> дисперсии {'равны' if equal_var else 'не равны'}"
        )
        # Применяем классический t-критерий Стьюдента (при equal_var=True) или критерий Уэлча (при equal_var=False)
        stat, p = stats.ttest_ind(x, y, equal_var=equal_var, alternative=alt)
        name = "t-критерий Стьюдента" if equal_var else "t-критерий Уэлча"
        stat_name = "t"
    else:
        # Для ненормальных выборок используем непараметрический U-критерий Манна–Уитни
        stat, p = stats.mannwhitneyu(x, y, alternative=alt)
        name, stat_name = "U-критерий Манна-Уитни", "U"

    print(f"Критерий: {name}; {stat_name} = {stat:.4f}; p-value = {p:.4f}")
    print(f"Решение (альфа = {ALPHA}): {decision(p)}")

# =====================================================================
# П.5. Динамика ЦФО: пара лет (зависимые выборки)
# =====================================================================
y1, y2 = YEAR_PAIR
print("\n" + "=" * 72)
print(f"П.5. Динамика {FO_NAME}: {y1} и {y2} г. (зависимые выборки)")
print("=" * 72)

a1 = get_sample(FO_NAME, y1)  # Данные ЦФО за первый год (2010)
a2 = get_sample(FO_NAME, y2)  # Данные ЦФО за второй год (2021)
diff = a2 - a1  # Вектор разностей по регионам (изменения за период)

# Проверка распределения парных разностей на нормальность (критерий Шапиро-Уилка)
p_d = stats.shapiro(diff)[1]
print(
    f"H0: существенных изменений нет (M(D) = 0);  H1: M(D) != 0, D = X{y2} -"
    f" X{y1}"
)
print(
    f"Шапиро-Уилк для разностей: p = {p_d:.4f} -> "
    f"{'нормальное' if p_d > ALPHA else 'не нормальное'}"
)

# Если разности распределены нормально — берем парный t-критерий Стьюдента, иначе — непараметрический критерий Уилкоксона
if p_d > ALPHA:
    stat, p = stats.ttest_rel(a2, a1)
    print(f"Парный t-критерий: t = {stat:.4f}; p-value = {p:.4f}")
else:
    stat, p = stats.wilcoxon(a2, a1)
    print(f"Критерий Уилкоксона: T = {stat:.4f}; p-value = {p:.4f}")

print(f"Решение (альфа = {ALPHA}): {decision(p)}")
print(
    f"Средняя разность = {diff.mean():.3f}, медиана разностей ="
    f" {np.median(diff):.3f}"
)

# =====================================================================
# П.6. ЦФО vs РФ в целом (одна выборка)
# =====================================================================
print("\n" + "=" * 72)
print(f"П.6. Сравнение {FO_NAME} с РФ (одновыборочный анализ)")
print("=" * 72)

for year in YEARS:
    s = get_sample(FO_NAME, year)  # Выборка по регионам ЦФО
    rf = get_rf(year)  # Фиксированное эталонное значение (среднее по всей РФ)

    # Задаем направление односторонней альтернативной гипотезы H1
    if normal_fo[year]:
        alt = "greater" if s.mean() > rf else "less"
    else:
        alt = "greater" if np.median(s) > rf else "less"
    sign = ">" if alt == "greater" else "<"

    print(f"\n--- {year} г. (РФ = {rf}) ---")
    print(f"H0: M({FO_NAME}) = {rf};  H1: M({FO_NAME}) {sign} {rf}")

    # При нормальности используем одновыборочный t-критерий, иначе — одновыборочный критерий Уилкоксона
    if normal_fo[year]:
        stat, p = stats.ttest_1samp(s, rf, alternative=alt)
        print(f"Одновыборочный t-критерий: t = {stat:.4f}; p-value = {p:.4f}")
    else:
        # Для Уилкоксона передается разность между выборкой и эталоном
        stat, p = stats.wilcoxon(s - rf, alternative=alt)
        print(f"Критерий Уилкоксона (1 выб.): T = {stat:.4f}; p-value = {p:.4f}")

    print(f"Решение (альфа = {ALPHA}): {decision(p)}")