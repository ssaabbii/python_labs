from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.metrics import accuracy_score


class RandomForest:
    def __init__(self, trees=50):
        self.n = trees  # количество деревьев
        self.models = []  # список для хранения моделей
        self.cols_for_model = []  # для хранения столбцов, связанных с каждой моделью
        self.df_pred = []  # для хранения данных всех моделей
        self.trained = False  # флаг

    # Training
    def fit(self, x_data, y_data):
        # для каждой модели
        for i in range(self.n):
            # рандомно выбираем два столбца
            cols = random.sample(list(x_data.columns), 2)
            self.cols_for_model.append(cols)
            # рандомно выбираем 50 строк
            indexes = random.sample(list(x_data.index), 50)
            # создаем training data
            x_train = x_data.iloc[indexes][cols]
            y_train = y_data[indexes]
            # инициализируем модель
            model = DecisionTreeClassifier()
            # обучаем, используя созданные данные
            model.fit(x_train, y_train)
            # добавляем созданную модель в список models
            self.models.append(model)
            self.trained = True

    def predict(self, x_test):
        # для каждой модели
        for i in range(len(self.models)):
            # Выбор столбцов, соответствующих каждой модели
            test_cols = self.cols_for_model[i]
            # Создание test data для это определенной модели
            test_data = x_test[test_cols]
            # предсказываем
            pred = self.models[i].predict(test_data)
            # создаем фрейм с предсказанными значениями
            pred = pd.DataFrame(pred)
            # Комбинируем все предсказания модели для создания большого фрейма
            if len(self.df_pred) == 0:
                self.df_pred = pred
            else:
                self.df_pred = pd.concat([self.df_pred, pred], axis=1)

        y_pred = []  # Stores the actual output
        # Наиболее встречающиеся в y_pred
        for i in range(len(self.df_pred)):
            y_pred.append(stats.mode(self.df_pred.iloc[i]).mode[0])

        return y_pred


def main():
    dataset = pd.read_csv('titanic_prepared.csv', index_col=0)
    x = dataset.drop('label', axis=1)  # на чем обучаемся
    y = dataset['label']  # целевой столбец

    test_ind = random.sample(list(dataset.index), 30)
    test = dataset.iloc[test_ind]

    x_test = test.drop('label', axis=1)  # на чем обучаемся
    y_test = test['label']  # целевой столбец

    # creating an instance
    model = RandomForest(trees=50)
    # Training the model
    model.fit(x, y)
    # Predicting 
    y_pred = model.predict(x_test)

    # Model Evaluation

    print("Accuracy :", accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    main()