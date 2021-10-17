def matrix_creation(file):
    """
    Reads the file and converts the read strings into matrix
    which is implemented using lists of lists

    Parameters
    ----------
    file: file
    file with input values(matrix)

    Returns
    ----------
    matrix0: matrix in the form of lists of lists

    """
    matrix0 = [list(map(int, row.split())) for row in file.readlines()]
    return matrix0


class Worker:

    def __init__(self, name):
        self.name = name
        self.bank = 0

    def take_salary(self, salary):
        """
        Increments the internal counter for each instance
        of the class by the value passed to it

        Parameters
        ----------
        salary: int
        value of worker's salary

        Returns
        ----------
        Increments balance of worker
        """

        self.bank += salary


class Operation:

    def summation(self, sum1, sum2):
        """
        Sums up 2 matrices

        Parameters
        ----------
        sum1: matrix in the form of lists of lists
        the first matrix
        sum2: matrix in the form of lists of lists
        the second matrix

        Returns
        ----------
        summ: matrix in the form of lists of lists
        sum of 2 matrices
        """

        summ = []
        for i in range(len(sum1)):
            value = []
            for j in range(len(sum1[0])):
                value.append(sum1[i][j] + sum2[i][j])
            summ.append(value)

        return summ

    def subtraction(self, minuend, subtrahend):
        """
        Subtracts the second matrix from the first

        Parameters
        ----------
        minuend: matrix in the form of lists of lists
        the first matrix
        subtrahend: matrix in the form of lists of lists
        the second matrix

        Returns
        ----------
        difference: matrix in the form of lists of lists
        difference of 2 matrices
        """

        difference = []
        for i in range(len(minuend)):
            value = []
            for j in range(len(minuend[0])):
                value.append(minuend[i][j] - subtrahend[i][j])
            difference.append(value)

        return difference


class Accountant:

    def __init__(self):
        pass

    def give_salary(self, worker, coefficient, work):
        """
        Сalls the method take_salary of instances of classes Pupa or Lupa

        Parameters
        ----------
        worker: an instance of the Pupa class or Lupa
        coefficient: int
        individual coefficient of worker
        work: list of lists
        number of rows in the work(result matrix)

        Returns
        ----------
        worker's salary
        """

        worker.take_salary(coefficient * len(work))


class Pupa(Worker):

    def do_work(self, filename1, filename2):
        """
        Reads from 2 files and sums up them elementwise

        Parameters
        ----------
        filename1: file
        the first matrix
        filename2: file
        the second matrix

        Returns
        ----------
        Matrix in the form of lists of lists
        sum of 2 matrices
        """

        input1 = open(filename1, "r")
        input2 = open(filename2, "r")
        return Operation().summation(matrix_creation(input1), matrix_creation(input2))


class Lupa(Worker):

    def do_work(self, filename1, filename2):
        """
        Reads from 2 files and subtract them elementwise

        Parameters
        ----------
        filename1: file
        the first matrix
        filename2: file
        the second matrix

        Returns
        ----------
        Matrix in the form of lists of lists
        sum of 2 matrices
        """

        input1 = open(filename1, "r")
        input2 = open(filename2, "r")
        return Operation().subtraction(matrix_creation(input1), matrix_creation(input2))


def main():
    p = Pupa('pupa')
    l = Lupa('lupa')
    acc = Accountant()
    a = p.do_work('matrix1.txt', 'matrix2.txt')
    b = l.do_work('matrix1.txt', 'matrix2.txt')
    print('Sum: ')
    for i in a:
        print(i)
    print('Sub: ')
    for i in b:
        print(i)
    acc.give_salary(p, 10000, a)
    acc.give_salary(l, 15000, b)
    print('Pupas salary - ', p.bank)
    print('Lupas salary - ', l.bank)


if __name__ == "__main__":
    main()
