
import pandas as pd
import numpy as np
import math

excel_data = pd.read_excel('.\lab3.xlsx')
data = pd.DataFrame(excel_data).to_numpy()
print("Матрица данных размером 58x10:\n", pd.DataFrame(data))

#excel_data = pd.read_excel('.\lab3t.xlsx')
#data = pd.DataFrame(excel_data).to_numpy()
#print("Матрица данных размером 5x2:\n", pd.DataFrame(data))


n = 58
p = 10

nt = 5
pt = 2

y = pd.DataFrame(data[:, 0]).to_numpy()
print("\tМатрица Y: \n", pd.DataFrame(y), "\n")

x = pd.DataFrame({'0': data[:, 1], '1': data[:, 2], '2': data[:, 3], '3': data[:, 4], '4': data[:, 5], '5': data[:, 6], '6': data[:, 7], '7': data[:, 8], '8': data[:, 9]}).to_numpy()    #вектор независимых переменных
print("\tМатрица X: \n", pd.DataFrame(x), "\n")


x1 = np.transpose(x)
x2 = np.dot(x1, x)
x2 = np.linalg.inv(x2)
x2 = np.dot(x2, x1)
alpha = np.dot(x2, y)
print("\tМНК-оценка вектора коэффициентов уравнения линейной множественной регрессии: \n", pd.DataFrame(alpha))

y_raschetnoe = np.dot(x, alpha)
print("\tРасчетное значение Y: \n", pd.DataFrame(y_raschetnoe), "\n")

y_sred_raschetnoe = y_raschetnoe.mean()
print("\tСреднее расчетное значение: \n", y_sred_raschetnoe, "\n")
    
y_fact_avg = y.mean()
print("\tСреднее фактическое значение: \n", np.average(y), "\n")

e = y - y_raschetnoe
print("\tВектор оценочных отклонений: \n", pd.DataFrame(e), "\n")

print(f"Уравнение регрессии: \n\ty = {alpha[0][0]}*x1 + {alpha[1][0]}*x2 + {alpha[2][0]}*x3 + {alpha[3][0]}*x4 + E\n")

sum_e = 0
sum_y = 0
for k in range(0, n):
    sum_e += e[k] * e [k]
    sum_y += (y[k] - y_fact_avg)**2
R = 1 - (sum_e / sum_y)

print("\tКоэффициент детерминации: \n", R[0], "\n")