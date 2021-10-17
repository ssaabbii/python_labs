# Pascal's Triangle
import argparse

parser = argparse.ArgumentParser(description='Process the integer and print the triangle')
parser.add_argument('N', type=int, help='Height of the triangle')
args = parser.parse_args()


# функция для вычисления факториала
def fact(n):
    x = 1
    for k in range(1, n + 1):
        x *= k
    return x


N = args.N

for i in range(0, N):
    # используется для пробелов слева ((N-i)*3 пробелов на каждой строке)
    print("   " * (N - i), end="")

    for j in range(0, i + 1):
        # nCk = n!/((n-k)!*k!) - биномиальный коэффициент, формула сочетания
        r = fact(i) // (fact(j) * fact(i - j))
        # str.center() позиционирует по центру str, дополняя её справа и слева до требуемой длины
        print(str(r).center(6), end="")
    # для перехода на следующую строку
    print()

