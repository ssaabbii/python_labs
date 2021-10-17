import numpy as np
import argparse


def the_first(a, b, p):
    """
        the first implementation option

        Parameters
        ----------
        a: np array
           real data
        b: np array
           syntactic data
        p: float
           probability

        Returns
        -------
        res : np array
            result array

    """
    res = (np.where(np.random.rand(len(a)) > p, a, b))
    print(res)


def the_second(a, b, p):
    """
        the second implementation option

        Parameters
        ----------
        a: np array
           real data
        b: np array
           syntactic data
        p: float
           probability

        Returns
        -------
        res : np array
            result array

    """
    r = np.random.choice([True, False], len(a), True, [1 - p, p])
    arr = np.array([True] * len(a))
    res = np.select([r, arr], [a, b])
    print(res)


def main():

    parser = argparse.ArgumentParser(description='Selection of random array elements')
    parser.add_argument('p', type=float, help='Probability')
    args = parser.parse_args()

    p = args.p

    a = np.loadtxt('file1.txt')
    b = np.loadtxt('file2.txt')

    the_first(a, b, p)

    the_second(a, b, p)


if __name__ == '__main__':
    main()
