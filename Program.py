import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, RobustScaler, StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet, SGDRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV

# Выгружаем данные из X_nup и X_bp в формат DataFrame
def load_data(path):
    df1 = pd.read_excel(path + r"\X_nup.xlsx")
    df2 = pd.read_excel(path + r"\X_bp.xlsx")

    dataset_inner = pd.merge(df1,df2,how = 'inner') # Объединяем наши таблицы в одну
    dataset_inner.drop('Unnamed: 0', axis = 1, inplace = True) #Удаляем второй столбец с индиксами

    return dataset_inner

# Проводим разведку данных
def intelligence_analysis(dataset_inner):

    #Составляем таблицу соотношения попарной корреляции
    data_corr = dataset_inner.corr()
    fig, axs = plt.subplots(figsize=(10, 7))
    sns.heatmap(data_corr, annot=True, fmt='.2f', cmap='PRGn', ax=axs, linewidths = 0.1)
    plt.show()

    #Смотрим процент пропущенных данных
    print()
    print("Процент пропущенных данных:")
    for col in dataset_inner.columns:
        pct_missing = np.mean(dataset_inner[col].isnull())
        print('{} - {}%'.format(col, round(pct_missing*100)))
    print()

    #Рисуем гистаграммы распределения каждой переменной
    fig = plt.figure(figsize=(8, 7))
    fig.tight_layout(h_pad = 3)
    for i, column in enumerate(dataset_inner.columns):
        ax = fig.add_subplot( 4, 4, i+1)
        sns.histplot(data=dataset_inner, x=column, kde=True, bins=35, color = 'blue')
        ax.set_ylabel('Количество записей')
    fig.tight_layout()
    plt.show()

    #Рисуем попарные графики рассеяния точек
    k = 0
    index = 0
    for i in range(0,12):
        fig, ax = plt.subplots(3, 4, figsize=(8, 7))
        fig.tight_layout(h_pad = 3)
        numbers = list(range(0,12))
        numbers.pop(i)
        for name, values in dataset_inner.iloc[:, np.r_[numbers]].items():
            if(k != 0 | k % 3 == 0):
                index += 1
                k = 0
            ax[k, index].scatter(x = values, y = dataset_inner.iloc[:,i].values)
            ax[k, index].set_xlabel(name)
            ax[k, index].set_ylabel(dataset_inner.iloc[:,i].name, fontsize = 7)
            k += 1
        k = 0
        index = 0
    plt.show()
    
    #Рисуем "ящик с усами" для наших данных
    dataset_box = dataset_inner.iloc[:, 0:13].values
    dataset_names = dataset_inner.iloc[:, 0:13].columns
    plt.boxplot(dataset_box, labels = dataset_names, vert = False)
    plt.show()

    #Выводим описание данных по каждому столбцу
    for name, values in dataset_inner.items():
        print(values.describe())
        print()
    
    #Смотрим столбцы в чьих значения повторяются больше чем 50%
    print("Столбцы с повторяющимеся данными >50%:")
    num_rows = len(dataset_inner.index)
    low_information_cols = []
    for col in dataset_inner.columns:
        cnts = dataset_inner[col].value_counts(dropna=False)
        top_pct = (cnts/num_rows).iloc[0]
    
        if top_pct > 0.5:
            low_information_cols.append(col)
            print('{0}: {1:.5f}%'.format(col, top_pct*100))
            print(cnts)
            print()

    # Рисуем график плотности распределения данных
    fig, axs = plt.subplots(figsize=(10, 7))
    dataset_inner.plot(kind='kde', ax=axs)
    plt.legend(dataset_names)
    plt.xlabel("Значение параметров")
    plt.ylabel("Плотность распределения данных")
    plt.show()

# Производим очищение данных от сомнительных записей 
def delete_excess_data(dataset_inner, coef):
    for name, values in dataset_inner.items():
        q75, q25 = np.percentile(dataset_inner.loc[:,name], [75,25])
        intr_qr = q75 - q25
        _max = q75 + (coef * intr_qr)
        _min = q25 - (coef * intr_qr)
        df_without_noise = dataset_inner.loc[(dataset_inner[name] < _max) & (dataset_inner[name] > _min)]
        dataset_inner = df_without_noise

    return dataset_inner

# Предварительная обработка тестовых и тренировачных данных
def preprocessing_data(dataset):
    columns = dataset.columns
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(np.array(dataset))
    dataset = pd.DataFrame(dataset, columns = columns)

    return dataset

# Проводим обучение несколькими методами и смотрим результаты 
def machine_learn(dataset_inner, number_result_columns):
    number_test_columns = list(range(0, dataset_inner.columns.size))
    if(type(number_result_columns) == list):
        for index in number_result_columns:
            number_test_columns.pop(index)
    else:
        number_test_columns.pop(number_result_columns)
    X = dataset_inner.iloc[:, np.r_[number_test_columns]].values
    y = dataset_inner.iloc[:, np.r_[number_result_columns]].values
    if(y.shape[1] == 1):
        y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=2)

    models_dir = [ExtraTreesRegressor(max_depth = 5), LinearRegression(n_jobs = -1), BayesianRidge(),
                  ElasticNet(random_state = 2), SVR(kernel = 'linear'), 
                  RandomForestRegressor(),
                  KNeighborsRegressor(), SGDRegressor()]
    models_names = ["ETR", "LR", "BR", "EN", "SVR", "RFR", "KNR", "SGDR"]
    df = {}

    for i in range(y_train.shape[1]):
        r2 = {}
        mae = {}
        for model in models_dir:
            m = str(model)
            model.fit(X_train, y_train[:,i])
            r2[m] = r2_score(y_test[:,i], model.predict(X_test))
            mae[m] = mean_absolute_error(y_test[:,i], model.predict(X_test))
        df[i] = pd.DataFrame(r2.items(), columns = [ '','R2_Y%s'%str(i + 1)])
        df[i + y_train.shape[1]] = pd.DataFrame(mae.items(), columns = [ '','MAE_Y%s'%str(i + 1)])
        
        print(df[i])
        print(df[i + y_train.shape[1]])

    fig = plt.figure(figsize=(8, 7))

    for i in range(y_train.shape[1]):
        ax = fig.add_subplot(2, y_train.shape[1], i + 1)
        ax.bar(df[0].iloc[:,0].tolist(), df[i].loc[:,'R2_Y%s'%str(i + 1)].tolist(), color = 'green')
        ax.set_xlabel('R2_Y%s'%str(i + 1))
        plt.xticks(df[0].iloc[:,0].tolist(), rotation='vertical', labels = models_names) 

        ax = fig.add_subplot(2, y_train.shape[1], i + 1 + y_train.shape[1])
        ax.bar(df[i + y_train.shape[1]].iloc[:,0].tolist(), df[i + y_train.shape[1]].loc[:,'MAE_Y%s'%str(i + 1)].tolist(),)
        ax.set_xlabel('MAE_Y%s'%str(i + 1))
        plt.xticks(df[i + y_train.shape[1]].iloc[:,0].tolist(), rotation='vertical', labels = models_names)
        fig.tight_layout(h_pad = 0.5)

    plt.show()
    print()

# Ищем гиперпараметры для наших методов
def grid_searchCV(dataset_inner, number_result_columns):
    number_test_columns = list(range(0, dataset_inner.columns.size))
    if(type(number_result_columns) == list):
        for index in number_result_columns:
            number_test_columns.pop(index)
    else:
        number_test_columns.pop(number_result_columns)
    X = dataset_inner.iloc[:, np.r_[number_test_columns]].values
    y = dataset_inner.iloc[:, np.r_[number_result_columns]].values
    if(y.shape[1] == 1):
        y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

    for i in range(y_train.shape[1]):
        grid_param_1 = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'epsilon': [0.01, 0.05, 0.1],
        }
        gd_sr = GridSearchCV(estimator = SVR(), param_grid=grid_param_1, cv=10, n_jobs=-1)
        gd_sr.fit(X_train, y_train[:,i])
        best_parameters = gd_sr.best_params_
        print("Гиперпараметры для : " + str(SVR()))
        print(best_parameters)

        grid_param_2 = {
            'random_state': [2, 10, 50],
            'max_iter': [100, 200, 450]

        }
        gd_sr = GridSearchCV(estimator = ElasticNet(), param_grid=grid_param_2, cv=10, n_jobs=-1)
        gd_sr.fit(X_train, y_train[:,i])
        best_parameters = gd_sr.best_params_
        print("Гиперпараметры для : " + str(ElasticNet()))
        print(best_parameters)

# Обучаем нейросеть прогнозировать данные
def neural_network_learn(dataset_inner, number_result_columns):
    number_test_columns = list(range(0, dataset_inner.columns.size))
    if(type(number_result_columns) == list):
        for index in number_result_columns:
            number_test_columns.pop(index)
    else:
        number_test_columns.pop(number_result_columns)
    X = dataset_inner.iloc[:, np.r_[number_test_columns]].values
    y = dataset_inner.iloc[:, np.r_[number_result_columns]].values

    if(y.shape[1] == 1):
        y.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    list_parameters = {}
    for i in range(y_train.shape[1]):
        regr = MLPRegressor(hidden_layer_sizes = (100, 100), random_state = 2).fit(X_train, y_train) # Используем 2 скрытых слоя по 100 нейронов
        r2 = r2_score(y_test[:,i], regr.predict(X_test))
        mae = mean_absolute_error(y_test[:,i], regr.predict(X_test))
        list_parameters[i] = ['R2_Y%s'%str(i + 1), 'MAE_Y%s'%str(i + 1)]
        list_parameters[i + y_train.shape[1]] = [r2, mae]
        
        print(list_parameters[i])
        print(list_parameters[i + y_train.shape[1]])

    fig = plt.figure(figsize=(3, 10))

    for i in range(y_train.shape[1]):
        ax = fig.add_subplot(2, y_train.shape[1], i + 1)
        ax.bar(list_parameters[i], list_parameters[i + y_train.shape[1]], )
        ax.set_xlabel('MLPR')
        plt.ylim(-0.5, 0.5)
        plt.xlim(-0.5, 1.5)

    plt.show()
    print()

def main():
    path = input("Введите путь к папке с Datasets: ") #Ввод пути к Datasets
    dataset_inner = load_data(path)

    print("Количество строк до очищения данных: %s"%str(dataset_inner.shape[0]))
    intelligence_analysis(dataset_inner)
    
    coef = float(input("Введите коэф. для удаления шумов: "))
    dataset_inner = delete_excess_data(dataset_inner, coef)
    print("Количество строк после очищения данных: %s"%str(dataset_inner.shape[0]))

    dataset_inner = preprocessing_data(dataset_inner)
    intelligence_analysis(dataset_inner)

    print(dataset_inner.columns.values)
    number_result_columns = list()
    count_result_columns = int(input("Введите кол-во прогнозирующих столбцов: "))
    for index in range(count_result_columns):
        number_column = int(input("Введите %sй номер столбца начиная с 0: "%str(index + 1)))
        number_result_columns.append(number_column)
    machine_learn(dataset_inner, number_result_columns)

    number_result_columns.clear()
    count_result_columns = int(input("Введите кол-во прогнозирующих столбцов: "))
    for index in range(count_result_columns):
        number_column = int(input("Введите %sй номер столбца начиная с 0: "%str(index + 1)))
        number_result_columns.append(number_column)
    grid_searchCV(dataset_inner, number_result_columns)

    number_result_columns.clear()
    count_result_columns = int(input("Введите кол-во прогнозирующих столбцов: "))
    for index in range(count_result_columns):
        number_column = int(input("Введите %sй номер столбца начиная с 0: "%str(index + 1)))
        number_result_columns.append(number_column)
    neural_network_learn(dataset_inner, number_column)

    input()
    exit()

if __name__ == "__main__":
	main()
