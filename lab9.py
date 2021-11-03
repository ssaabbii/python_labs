import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np


def accuracy(X_train, y_train, X_test, y_test, model, text):
    """
              the function outputs the proportion of correct answers

              Parameters
              ----------
              X_train: dataframe
                 train dataset without "label"
              y_train: dataframe
                 train dataset consisting of a column "label"
              X_test: dataframe
                 test dataset without "label"
              y_test: dataframe
                 test dataset consisting of a column "label"
              model: classifier
                 the model that is trained
              text: str
                 type of model training
              Returns
              -------
              the proportion of correct answers

     """

    print(text)
    print("Accuracy on training set: {:.3f}".format(model.score(X_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(model.score(X_test, y_test)))
    print()


def training(X_train, y_train, X_test, y_test, dataset):
    """
                      the function trains the model to predict the label column from the rest of the table columns

                      Parameters
                      ----------
                      X_train: dataframe
                         train dataset without "label"
                      y_train: dataframe
                         train dataset consisting of a column "label"
                      X_test: dataframe
                         test dataset without "label"
                      y_test: dataframe
                         test dataset consisting of a column "label"
                      dataset: dataframe
                         whole dataset
                      Returns
                      -------
                      models that predict the 'label' column by the remaining columns of the table

    """
    # DECISION TREE
    clf = DecisionTreeClassifier(max_depth=4, criterion='entropy')  # создаем классификатор
    clf = clf.fit(X_train, y_train)  # обучаем
    _ = clf.predict(X_test)  # предсказываем
    # tree.plot_tree(clf, feature_names=X_train.columns, filled=True) # вывод дерева
    # plt.show()
    accuracy(X_train, y_train, X_test, y_test, clf, "Decision Tree")  # доля правильных ответов

    # XGBoost
    model_xgboost = XGBClassifier(n_estimators=20, max_depth=4)
    model_xgboost.fit(X_train, y_train)
    _ = model_xgboost.predict(X_test)
    accuracy(X_train, y_train, X_test, y_test, model_xgboost, "XGBoost")

    # Logistic Regression
    model = LogisticRegression(C=0.1, solver='lbfgs')
    model.fit(X_train, y_train)
    _ = model.predict(X_test)
    accuracy(X_train, y_train, X_test, y_test, model, "Logistic Regression")

    # С помощью Decision Tree выбрать 2 самых важных признака и проверить точность модели, обученной только на них.
    importances = clf.feature_importances_
    features = X_train.columns
    # Добавление сортировки по важности
    indices = np.argsort(importances)
    plt.title('Важность признаков')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), features[indices])
    plt.xlabel('Относительная важность')
    plt.show()

    X1 = dataset.filter(['day', 'morning'], axis=1)  # на чем обучаемся
    y1 = dataset['label']  # целевой столбец

    # разделяем dataset на выборку train/test
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.1,
                                                            random_state=1)  # 90% train, 10% test

    clf2 = DecisionTreeClassifier(max_depth=4, criterion='entropy')  # создаем классификатор
    clf2 = clf2.fit(X_train1, y_train1)  # обучаем
    _ = clf2.predict(X_test1)   # предсказываем
    accuracy(X_train1, y_train1, X_test1, y_test1, clf2, "Decision Tree (the 2 most important signs)")


def main():
    # 1. Загрузить файл, разделить его на train и test. Для test взять 10% случайно выбранных строк таблицы.
    dataset = pd.read_csv('titanic_prepared.csv', index_col=0)

    X = dataset.drop('label', axis=1)  # на чем обучаемся
    y = dataset['label']  # целевой столбец

    # разделяем dataset на выборку train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)  # 90% train, 10% test

    # 2-5. Обучить модели: Decision Tree, XGBoost, Logistic Regression из библиотек sklearn и xgboost.
    # Обучить модели предсказывать столбец label по остальным столбцам таблицы. + Accuracy
    # Точности всех моделей не должны быть ниже 85%
    # С помощью Decision Tree выбрать 2 самых важных признака и проверить точность модели, обученной только на них.

    training(X_train, y_train, X_test, y_test, dataset)
    '''
    model = RandomForestClassifier(n_estimators=20, max_depth=5, criterion='entropy')
    model.fit(X_train, y_train)
    _ = model.predict(X_test)
    print("Accuracy on training set: {:.3f}".format(model.score(X_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(model.score(X_test, y_test)))
    '''


if __name__ == '__main__':
    main()
