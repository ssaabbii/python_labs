import pandas as pd


def func(row):
    """
           This function:
           1)  divides the column (age) into 3 groups: children (under 18), adults (18 - 50), elderly (50+)
           2) changes column 'drink' into 1 or 0 depending on whether it was beer or not
           3) divides the column () into 3 groups: morning, day, evening

           Param:
                 row
           Return:
                 row
    """
    # 1.
    n = row.at['age']
    if 0 < n < 18:
        n = 'age_0'
    elif 18 <= n < 50:
        n = 'age_18'
    elif n >= 50:
        n = 'age_50'
    row.at[n] = 1

    # 2.
    n = row.at['drink']
    if 'beer' in n:
        row.at['drink'] = 1
    else:
        row.at['drink'] = 0

    # 3.
    n = row.at['session_start']
    t = ''
    i = 0
    q = n.hour
    if 5 < q < 12:
        t = 'morning'
    elif 12 < q < 17:
        t = 'day'
    elif 17 < q < 24:
        t = 'evening'
    else:
        t = 'morning'

    row.at[t] = 1

    return row


def main():
    df = pd.read_csv("cinema_sessions (1).csv", sep=' ', index_col=0)
    df2 = pd.read_csv("titanic_with_labels (1).csv", sep=' ', index_col=0)
    df = df.merge(df2, on='check_number')
    df['age_0'] = 0
    df['age_18'] = 0
    df['age_50'] = 0
    df['morning'] = 0
    df['day'] = 0
    df['evening'] = 0
    # преобразование в тип datetime
    df['session_start'] = pd.to_datetime(df['session_start'], errors='coerce')
    df = df.apply(func, axis=1)
    del df['age']
    df.to_csv("new.csv")
    print(df)


if __name__ == '__main__':
    main()