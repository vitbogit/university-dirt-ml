import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

#from sklearn.metrics import mean_squared_error
#from sklearn.inspection import PartialDependenceDisplay
#import matplotlib.backends.backend_pdf
#from sklearn.feature_selection import mutual_info_regression
#from xgboost import XGBRegressor
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('data.csv', index_col="fid")  # Считывание всей даты

data = data.drop(columns=['fid2'])  # Удаляем лишний столбец (второй fid)
data = data.drop(columns=['Растительность'])

# Варианты того что будем предсказывать
Y = ['Р2О5', 'К2О', 'Гидролитическая кислотность', 'рН водный', 'Гумус', 'рН солевой']

# Пользователь выбирает, что будет предсказываться:
print("\nВыберите, на что хотите получить прогноз:")
for i in range(len(Y)):
    print(i, ' - ', Y[i])
chosen_y_index = int(input())
chosen_y = Y[chosen_y_index]

# Фичи
features = ['1 канал', '2 канал', '3 канал', '4 канал',
            '5 канал', '6 канал', '7 канал', '8 канал']
# Не используются 'red color', 'green color', 'blue color'

# Колонка целевая
y = data.loc[:, chosen_y]

# Из чего предсказываем
x = data[features].copy()

# Разделяем данные, чтобы часть пошла на обучение модели, а часть на тестирование:
x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=0)

# Выбор модели
#model = RandomForestRegressor(random_state=1, n_estimators=15)
#model = XGBRegressor()
model = GradientBoostingRegressor(learning_rate=0.150, n_estimators=80)

# Обучение модели
model.fit(x_train, y_train)

# Построение предсказания
preds = model.predict(x_valid)

# Расчет отклонения
mae = mean_absolute_error(y_valid, preds)
print("\nСреднее отклонение =", mae)

# Заполняем графики
figure, axis = plt.subplots(2, 4)
for i in range(2):
    for j in range(4):

        # Точки на графики
        ind = i*4+j
        x_f = x_valid.loc[:, features[ind]]
        axis[i, j].scatter(x_f, y_valid, s=25, color='gray')
        axis[i, j].plot(x_train.loc[:, features[ind]], y_train, "1", color='b')
        axis[i, j].scatter(x_f, preds, s=70, facecolor='none', edgecolors='r')
        axis[i, j].set_xlabel(features[ind])
        axis[i, j].set_ylabel(y.name)
        axis[i, j].legend(['Целевые значения', 'Обучение', 'Предсказание'])

        # Усредняющая кривая (усредняет предсказания)
        np_x_f = x_f.to_numpy()
        p = np_x_f.argsort()
        np_x_f = np_x_f[p]
        np_preds = preds[p]
        fit = np.polyfit(np_x_f, np_preds, 10)
        pf = np.poly1d(fit)
        axis[i, j].plot(np_x_f, pf(np_x_f), 'brown')

figure.suptitle("Каналы спутника -> " + chosen_y + " (средн. отклн " + str(mae)[:4] + ")")

plt.show()
