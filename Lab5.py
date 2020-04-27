import random
import math
from _pydecimal import Decimal
from scipy.stats import f, t, ttest_ind, norm
from functools import reduce
from itertools import compress
import numpy as np
import time

N = 15
m = 3
p = 0.95
l = 1.215

min_x1, max_x1, min_x2, max_x2, min_x3, max_x3 = -3, 7, 0, 9, -8, 10
mean_Xmin = (min_x1 + min_x2 + min_x3) / 3
mean_Xmax = (max_x1 + max_x2 + max_x3) / 3

min_y = 200 + mean_Xmin
max_y = 200 + mean_Xmax


def gen_matrix(array):
    return [row + [row[0] * row[1], row[0] * row[2], row[1] * row[2], \
                   row[0] * row[1] * row[2]] + list(map(lambda x: round(x ** 2, 5), \
                                                        row)) for row in array]


def cochran_criterion(m, N, y_matrix, p=0.95):
    print("\nКритерій Кохрена: ")
    yVariance = [np.var(i) for i in y_matrix]  # пошук дисперсії за допомогою numpy
    yVar_max = max(yVariance)
    Gp = yVar_max / sum(yVariance)
    f1 = m - 1
    f2 = N
    q = 1 - p
    Gt = cochran_value(f1, f2, q)
    print(f"Gp = {Gp:.3f}; Gt = {Gt:.3f}; f1 = {f1}; f2 = {f2}; q = {q:.3f}")
    if Gp < Gt:
        print("Gp < Gt => Дисперсія однорідна.")
        return True
    else:
        print("Gp > Gt => Дисперсія неоднорідна.")
        return False


def x_i(i, x_coded):
    try:
        assert i <= 10
    except:
        raise AssertionError("Помилка: і > 10")
    with_null_x = list(map(lambda x: [1] + x, gen_matrix(x_coded)))
    res = [row[i] for row in with_null_x]
    return np.array(res)


def m_ij(*arrays):
    return np.average(reduce(lambda accum, el: accum * el, arrays))


def calculate_yTheor(x_table, b_coefficients, importance):
    x_table = [list(compress(row, importance)) for row in x_table]
    b_coefficients = list(compress(b_coefficients, importance))
    y_vals = np.array([sum(map(lambda x, b: x * b, row, b_coefficients)) for row in x_table])
    return y_vals


def cochran_value(f1, f2, q):
    partResult1 = q / f2  # (f2 - 1)
    params = [partResult1, f1, (f2 - 1) * f1]
    fisher = f.isf(*params)
    result = fisher / (fisher + (f2 - 1))
    return Decimal(result).quantize(Decimal('.0001'))


def student_value(f3, q):
    return Decimal(abs(t.ppf(q / 2, f3))).quantize(Decimal('.0001'))


def fisher_value(f3, f4, q):
    return Decimal(abs(f.isf(q, f4, f3))).quantize(Decimal('.0001'))


x0 = [(max_x1 + min_x1) / 2, (max_x2 + min_x2) / 2, (max_x3 + min_x3) / 2]
detx = [abs(min_x1 - x0[0]), abs(min_x2 - x0[1]), abs(min_x3 - x0[2])]


# нормовані значення факторів
x_coded = [[-1, -1, -1],
           [-1, +1, +1],
           [+1, -1, +1],
           [+1, +1, -1],
           [-1, -1, +1],
           [-1, +1, -1],
           [+1, -1, -1],
           [+1, +1, +1],
           [-1.215, 0, 0],
           [+1.215, 0, 0],
           [0, -1.215, 0],
           [0, +1.215, 0],
           [0, 0, -1.215],
           [0, 0, +1.215],
           [0, 0, 0]]

# натуралізовані значення факторів
x_natur = [[min_x1, min_x2, min_x3],
           [min_x1, max_x2, max_x3],
           [max_x1, min_x2, max_x3],
           [max_x1, max_x2, min_x3],

           [min_x1, min_x2, max_x3],
           [min_x1, max_x2, min_x3],
           [max_x1, min_x2, min_x3],
           [max_x1, max_x2, max_x3],
           [-l * detx[0] + x0[0], x0[1], x0[2]],
           [l * detx[0] + x0[0], x0[1], x0[2]],
           [x0[0], -l * detx[1] + x0[1], x0[2]],
           [x0[0], l * detx[1] + x0[1], x0[2]],
           [x0[0], x0[1], -l * detx[2] + x0[2]],
           [x0[0], x0[1], l * detx[2] + x0[2]],
           [x0[0], x0[1], x0[2]]]

duration = 0
for i in range(100):
    start = time.time()
    xMatrix_coded = gen_matrix(x_coded)
    print("Кодовані фактори:")
    for row in xMatrix_coded:
        print(row)
    
    xMatrix_natur = gen_matrix(x_natur)
    with_null_x = list(map(lambda x: [1] + x, xMatrix_natur))
    
    y_values = [[random.random() * (max_y - min_y) + min_y for i in range(m)] for j in range(N)]
    
    # Критерій Кохрена (Перша статистична перевірка)
    while not cochran_criterion(m, N, y_values):
        m += 1
        y_values = [[random.random() * (max_y - min_y) + min_y for i in range(m)] for j in range(N)]
    
    y_i = np.array([np.average(row) for row in y_values])
    coefficients = [[m_ij(x_i(column, x_coded) * x_i(row, x_coded)) for column in range(11)] for row in
                    range(11)]
    free_values = [m_ij(y_i, x_i(i, x_coded)) for i in range(11)]
    beta_coef = np.linalg.solve(coefficients, free_values)
    
    # Критерій Стьюдента (Друга статистична перевірка)
    y_matrix = y_values
    f3 = (m - 1) * N
    q = 1 - p
    T = student_value(f3, q)
    print("\nКритерій Стьюдента:")
    
    # Оцінка генеральної дисперсії відтворюваності
    Sb = np.average(list(map(np.var, y_matrix)))
    meanY = np.array(list(map(np.average, y_matrix)))
    Sbs_2 = Sb / (N * m)
    Sbs = math.sqrt(Sbs_2)
    x_vals = [x_i(i, x_coded) for i in range(11)]
    t_i = np.array([abs(beta_coef[i]) / Sbs for i in range(len(beta_coef))])
    
    # Перевірка значущості коефіцієнтів
    importance = [True if el > T else False for el in list(t_i)]
    
    x_i_names = list(compress(["1", "x1", "x2", "x3", "x12", "x13", "x23", "x123", "x1^2", "x2^2", "x3^2"], importance))
    betas = list(compress(beta_coef, importance))  # якщо importance=False => значення beta_coef не ввійде в betas
    
    # Пошук нового рівняння регресії
    print("Рівняння регресії без незначимих членів: y = ", end="")
    for i in range(len(betas)):
        print(f" {betas[i]:+.3f}*{x_i_names[i]}", end="")
    
    # Критерій Фішера (Третя статистична перевірка)
    d = len(list(filter(None, importance)))
    y_matrix = y_values
    f3 = (m - 1) * N
    f4 = N - d
    q = 1 - p
    print("\n\nКритерій Фішера:")
    
    yTheor = calculate_yTheor(xMatrix_natur, beta_coef, importance)
    meanY = np.array(list(map(np.average, y_matrix)))
    # Дисперсія адекватності
    Sad = m / (N - d) * (sum((yTheor - meanY) ** 2))
    yVariance = np.array(list(map(np.var, y_matrix)))
    s_v = np.average(yVariance)
    Fp = float(Sad / s_v)
    Ft = fisher_value(f3, f4, q)
    print(f"Fp = {Fp:.3f}, Ft = {Ft:.3f}", "\nFp < Ft => модель адекватна" if Fp < Ft else "Fp > Ft => модель неадекватна\n\n")
    stop = time.time()
    duration += stop - start
print ("В середньому одне проходження триває", duration/100)
    
