import pandas as pd

def change_sex(row):
    """
            the function changes male to 0 and female to 1

            Parameters
            ----------
            row

            Returns
            -------
            1 if row['sex'] is female
            0 if row['sex'] is male

    """

    if row['sex'] == 'Ж' or row['sex'] == 'ж':
        return 1
    else:
        if row['sex'] == 'М' or row['sex'] == 'м' or row['sex'] == 'M':
            return 0


def main():

    data = pd.read_csv('cinema_sessions (1).csv', sep=' ')
    file2 = pd.read_csv('titanic_with_labels.csv', sep=' ')

    # 1. Пол (sex): отфильтровать строки, где пол не указан, преобразовать оставшиеся в число 0/1;

    data1 = data[(data['sex'] == 'м') | (data['sex'] == 'М') | (data['sex'] == 'M') | (data['sex'] == 'Ж') | (
                data['sex'] == 'ж')]

    data1 = data1.apply(change_sex, axis=1)
    print(data1.head())

    # 2. Номер ряда в зале (row_number): заполнить вместо NAN максимальным значением ряда;

    max_row_number = data['row_number'].max()

    data2 = data['row_number'].fillna(max_row_number)

    print(data2.head())

    # 3. Количество выпитого в литрах (liters_drunk): отфильтровать отрицательные значения и нереально
    # большие значения (выбросы). Вместо них заполнить средним.

    data3 = data[(data['liters_drunk'] >= 0) & (data['liters_drunk'] <= 3)]
    mean = data3['liters_drunk'].mean()
    m = int(round(mean))
    data.loc[(data.liters_drunk < 0), 'liters_drunk'] = m
    data.loc[(data.liters_drunk > 3), 'liters_drunk'] = m

    print(data.head())


if __name__ == '__main__':
    main()