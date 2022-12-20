"""
Author: Arkadii Semenov

Description:
Create a precise calculation algorithm for matrix
operations with rational coefficients.
"""

import unittest


class Rational:
    """Class implementing operations for rational numbers"""

    def __init__(self, a:int, b:int):
        gcd = self._gcd(abs(a), abs(b))
        self.a = a // gcd
        self.b = b // gcd
        if self.b < 0:
            self.a *= -1
            self.b *= -1

    def __eq__(self, other):
        return self.a == other.a and self.b == other.b

    def __neg__(self):
        return self.__class__(-self.a, self.b)

    def __add__(self, other):
        new_a = self.a * other.b + other.a * self.b
        new_b = self.b * other.b
        return self.__class__(new_a, new_b)

    def __iadd__(self, other):
        res = self.__add__(other)
        self.a = res.a
        self.b = res.b

    def __sub__(self, other):
        return self + (-other)

    def __isub__(self, other):
        res = self.__sub__(other)
        self.a = res.a
        self.b = res.b

    def __mul__(self, other):
        new_a = self.a * other.a
        new_b = self.b * other.b
        return self.__class__(new_a, new_b)

    def __imul__(self, other):
        res = self.__mul__(other)
        self.a = res.a
        self.b = res.b

    def __truediv__(self, other):
        new_a = self.a * other.b
        new_b = self.b * other.a
        return self.__class__(new_a, new_b)

    def __itruediv__(self, other):
        res = self.__truediv__(other)
        self.a = res.a
        self.b = res.b

    def __repr__(self):
        if self.a == 0 or self.b == 1:
            return f'{self.a}'
        return f'{self.a}/{self.b}'

    def _gcd(self, a, b):
        """Find GCD using Euclid's algorithm"""
        if b > 0:
            if a == b: return a
            return self._gcd(b, a % b)
        return a

    def _lcm(self, a, b):
        """Find LCM using Euclid's algorithm"""
        return a * b // self._gcd(a, b)

    def gcd(self, other):
        return self.__class__(self._gcd(abs(self.a), abs(other.a)),
            self._lcm(abs(self.b), abs(other.b)))


class Matrix:
    """Class implementing matrix operations"""

    def __init__(self, height:int, width:int, values=None):
        self.width = width
        self.height = height
        self.values = [None] * (self.width * self.height)

        if values is not None:
            self.set_values(values)
        else:
            self.values = [Rational(0, 1) for i in range(self.height * self.width)]

    def set_value(self, i:int, j:int, value):
        """Set value to specified position of the matrix"""
        if not (0<=i<self.height and 0<=j<self.width):
            raise IndexError(f'Bad indexes for the matrix \
                {self.height}x{self.width}: {i} and {j}')

        if isinstance(value, int):
            self.values[i * self.width + j] = Rational(value, 1)
        elif isinstance(value, Rational):
            self.values[i * self.width + j] = value
        else:
            raise ValueError(f'Bad value type for matrix: {type(value)}')

    def set_values(self, arr):
        """Set values to specified position of the matrix"""

        if len(arr) != self.height * self.width:
            raise ValueError('Bad array length.')

        if isinstance(arr[0], int):
            for i in range(self.height * self.width):
                self.values[i] = Rational(arr[i], 1)
        elif isinstance(arr[0], Rational):
            for i in range(self.height * self.width):
                self.values[i] = arr[i]


    def get_value(self, i:int, j:int):
        """
        get_value(self, height_index, width_index)
        Returns walue with specified indexes
        """
        if not (0<=i<self.height and 0<=j<self.width):
            raise IndexError(f'Bad indexes for the matrix \
                {self.width}x{self.height}: {i} and {j}')

        return self.values[i*self.width + j]

    def __eq__(self, other):
        if self.width != other.width:
            return False

        if self.height != other.height:
            return False

        for i in range(self.height):
            for j in range(self.width):
                if self.get_value(i, j) != other.get_value(i, j):
                    return False
        return True

    def __add__(self, other):
        if self.width != other.width or self.height != other.height:
            raise ValueError(f'Cannot add matrices with different sizes:\
                {self.width}x{self.height} and {other.width}x{other.height}')

        result = self.__class__(self.height, self.width)

        for i in range(self.height):
            for j in range(self.width):
                result.set_value(i, j,
                    self.get_value(i, j) + other.get_value(i, j))

        return result

    def __neg__(self):
        result = self.__class__(self.height, self.width)
        for i in range(self.height):
            for j in range(self.width):
                result.set_value(i, j, -self.get_value(i, j))
        return result

    def __sub__(self, other):
        if self.width != other.width or self.height != other.height:
            raise ValueError(f'Cannot add matrices with different sizes:\
                {self.width}x{self.height} and {other.width}x{other.height}')

        result = self.__class__(self.height, self.width)

        for i in range(self.height):
            for j in range(self.width):
                result.set_value(i, j,
                    self.get_value(i, j) - other.get_value(i, j))

        return result

    def __rmul__(self, other):
        if isinstance(other, Rational):
            result = self.__class__(self.height, self.width)
            for i in range(self.height):
                for j in range(self.width):
                    result.set_value(i, j, other * self.get_value(i, j))
            return result

        if isinstance(other, int):
            result = self.__class__(self.height, self.width)
            for i in range(self.height):
                for j in range(self.width):
                    result.set_value(i, j, Rational(other, 1) * self.get_value(i, j))
            return result

        return self.__mul__(other)

    def __mul__(self, other):
        if isinstance(other, Matrix):
            if self.width != other.height:
                raise ValueError(
                    f'Bad dimensions for matrix multiplication \
                    {self.height}x{self.width} and {other.height}x{other.width}')

            result = self.__class__(self.height, other.width)

            for i in range(self.height):
                for j in range(other.width):
                    c = Rational(0, 1)
                    for k in range(self.width):
                        c = c + self.get_value(i, k) * other.get_value(k, j)
                    result.set_value(i, j, c)
            return result

        raise ValueError(f'Bad values: can multipliy \
            {type(self)} and {type(other)}')

    def __repr__(self):
        result = "Matrix(\n"
        for i in range(self.height):
            for j in range(self.width):
                result += str(self.get_value(i, j)) + ' '
            result += '\n'

        return result + ')'

    def add_row(self, from_index, to_index, multiplier):
        """Adds one row to another mutiplied by a coefficient"""
        for i in range(self.width):
            self.set_value(to_index, i,
                self.get_value(to_index, i) +\
                self.get_value(from_index, i) * multiplier)

    def mul_row(self, row_index, multiplier):
        """Multiplies the row by a coefficient"""
        for i in range(self.width):
            self.set_value(row_index, i, self.get_value(row_index, i) * multiplier)

    def transpose(self):
        """Returns transpose of the matrix"""
        result = self.__class__(self.width, self.height)
        for i in range(self.height):
            for j in range(self.width):
                result.set_value(j, i, self.get_value(i, j))
        return result

    def inverse(self):
        """
        Returns the inverse of the matrix,
        throws an exception if determinant is zero
        """
        if self.height != self.width:
            raise ValueError(f'Can\'t get inverse of non-square matrix: \
                {self.height}x{self.width}')

        copy = Matrix(self.height, self.width, values=self.values)
        result = self.__class__.unit(self.width)

        # Create a checking fo rthe matrix
        # for every row
        for i in range(self.height):
            if copy.get_value(i, i) == Rational(0, 1):
                # if there is a zero on the main diagonal
                for j in range(self.height):
                    if i != j and copy.get_value(j, i) != Rational(0, 1):
                        # add some other row to it with non-zero element
                        copy.add_row(j, i, Rational(1, 1))
                        result.add_row(j, i, Rational(1, 1))
                        break
                else:
                    # if all elements are zero -> det A = 0, throw an error
                    raise ZeroDivisionError('Bad matrix, determinant is zero')

        # proceed row coefficients up to down
        try:
            for col in range(self.height):
                main_row = col
                for add_row in range(self.height):
                    if add_row != main_row:
                        c = - copy.get_value(add_row, col) / copy.get_value(main_row, col)
                        copy.add_row(main_row, add_row, c)
                        result.add_row(main_row, add_row, c)
                c = Rational(1, 1) / copy.get_value(main_row, main_row)

                copy.mul_row(main_row, c)
                result.mul_row(main_row, c)
        except ZeroDivisionError:
            raise ValueError('Bad matrix, determinant is zero')

        for i in range(self.height):
            if copy.get_value(i, i) == Rational(0, 1):
                raise ValueError('Bad matrix, determinant is zero')
        return result

    @classmethod
    def unit(cls, size):
        """Returns a unit matrix n x n of the specified size"""
        return cls(size, size,
            values=[1 if i%(size+1) == 0 else 0 for i in range(size * size)])


class SLE:
    def __init__(self, a_values: Matrix, b_values: Matrix):
        if a_values.height != b_values.height:
            raise ValueError('bad input for SLE coefficients, not enough values')

        self.a_values = a_values
        self.b_values = b_values

        self.eqs = self.a_values.height
        self.vars = self.a_values.width

        # maximum rank of a SLE coefficient matrix
        self.mx_r = min(self.eqs, self.vars)

    def __repr__(self):
        print('SLE')
        print(self.a_values)
        print(self.b_values)

    def solve_self(self):
        # Create a checking fo rthe matrix
        # for every row
        for i in range(self.mx_r):
            if self.a_values.get_value(i, i) == Rational(0, 1):
                # if there is a zero on the main diagonal
                for j in range(height):
                    if i != j and self.a_values.get_value(j, i) != Rational(0, 1):
                        # add some other row to it with non-zero element
                        self.a_values.add_row(j, i, Rational(1, 1))
                        self.b_values.add_row(j, i, Rational(1, 1))

        print(self)


class TestRational(unittest.TestCase):
    """Test class for Rational"""

    def setUp(self):
        self.x1 = Rational(1, 2)
        self.x2 = Rational(2, 1)
        self.x3 = Rational(-1, 3)
        self.x4 = Rational(1, 3)
        self.x5 = Rational(14, 3)

    def test_gcd(self):
        self.assertEqual(self.x2.gcd(self.x5), Rational(2, 3))


    def test_add(self):
        self.assertEqual(self.x1 + self.x1, Rational(1, 1))
        self.assertEqual(self.x1 + self.x2, Rational(5, 2))
        self.assertEqual(self.x1 + self.x3, Rational(1, 6))
        self.assertEqual(self.x1 + self.x4, Rational(5, 6))
        self.assertEqual(self.x2 + self.x2, Rational(4, 1))
        self.assertEqual(self.x2 + self.x3, Rational(5, 3))
        self.assertEqual(self.x2 + self.x4, Rational(7, 3))
        self.assertEqual(self.x3 + self.x3, Rational(-2, 3))
        self.assertEqual(self.x3 + self.x4, Rational(0, 1))
        self.assertEqual(self.x4 + self.x4, Rational(2, 3))

    def test_min(self):
        self.assertEqual(self.x1 - self.x1, Rational(0, 1))
        self.assertEqual(self.x1 - self.x2, Rational(-3, 2))
        self.assertEqual(self.x1 - self.x3, Rational(5, 6))
        self.assertEqual(self.x1 - self.x4, Rational(1, 6))
        self.assertEqual(self.x2 - self.x2, Rational(0, 1))
        self.assertEqual(self.x2 - self.x3, Rational(7, 3))
        self.assertEqual(self.x2 - self.x4, Rational(5, 3))
        self.assertEqual(self.x3 - self.x3, Rational(0, 1))
        self.assertEqual(self.x3 - self.x4, Rational(-2, 3))
        self.assertEqual(self.x4 - self.x4, Rational(0, 1))

    def test_mul(self):
        self.assertEqual(self.x1 * self.x1, Rational(1, 4))
        self.assertEqual(self.x1 * self.x2, Rational(1, 1))
        self.assertEqual(self.x1 * self.x3, Rational(-1, 6))
        self.assertEqual(self.x1 * self.x4, Rational(1, 6))
        self.assertEqual(self.x2 * self.x2, Rational(4, 1))
        self.assertEqual(self.x2 * self.x3, Rational(-2, 3))
        self.assertEqual(self.x2 * self.x4, Rational(2, 3))
        self.assertEqual(self.x3 * self.x3, Rational(1, 9))
        self.assertEqual(self.x3 * self.x4, Rational(-1, 9))
        self.assertEqual(self.x4 * self.x4, Rational(1, 9))

    def test_div(self):
        self.assertEqual(self.x1 / self.x1, Rational(1, 1))
        self.assertEqual(self.x1 / self.x2, Rational(1, 4))
        self.assertEqual(self.x1 / self.x3, Rational(-3, 2))
        self.assertEqual(self.x1 / self.x4, Rational(3, 2))
        self.assertEqual(self.x2 / self.x2, Rational(1, 1))
        self.assertEqual(self.x2 / self.x3, Rational(-6, 1))
        self.assertEqual(self.x2 / self.x4, Rational(6, 1))
        self.assertEqual(self.x3 / self.x3, Rational(1, 1))
        self.assertEqual(self.x3 / self.x4, Rational(-1, 1))
        self.assertEqual(self.x4 / self.x4, Rational(1, 1))

    def test_rat_gcd(self):
        self.assertEqual(self.x2.gcd(self.x5), Rational(2, 3))


class TestMatrix(unittest.TestCase):
    """Tests matrix related functionality"""
    def setUp(self):
        self.mat_1 = Matrix(2, 2, values=[1, 2, 3, 4])
        self.mat_2 = Matrix(2, 2, values=[0, -2, 1, -4])
        self.mat_3 = Matrix(2, 2, values=[1, 0, 0, 1])
        self.mat_4 = Matrix(2, 2, values=[0, 1, 1, 0])
        self.mat_5 = Matrix(2, 1, values=[0, 1])
        self.mat_6 = Matrix(1, 3, values=[0, 0, 1])

        self.mat_a = Matrix(2, 1, values=[0, -3])
        self.mat_b = Matrix(4, 3, values=[1, 0, -1, 3, 5, 0, 6, 0, 2, 0, 2, 3])
        self.mat_c = Matrix(3, 2, values=[1, 2, 0, 1, 1, 0])
        self.mat_d = Matrix(3, 3, values=[
            0, 0, 1,
            0, 1, 0,
            1, 0, 0])
        self.mat_e = Matrix(5, 5, values=[
            1, -2, 0, 3, 6,
            12, 7, 0, 0, 0,
            0, -5, 0, -8, 5,
            3, 9, 1, 1, -7,
            0, 0, -6, 4, 2])

        self.mat_1p2 = Matrix(2, 2, values=[1, 0, 4, 0])
        self.mat_1p3 = Matrix(2, 2, values=[2, 2, 3, 5])
        self.mat_1p4 = Matrix(2, 2, values=[1, 3, 4, 4])

        self.mat_1m2 = Matrix(2, 2, values=[1, 0, 4, 0])

        self.mat_1s2 = Matrix(2, 2, values=[2, 4, 6, 8])

    def test_get(self):
        self.assertEqual(self.mat_1.get_value(0,0), self.mat_1.values[0])
        self.assertEqual(self.mat_1.get_value(0,1), self.mat_1.values[1])
        self.assertEqual(self.mat_1.get_value(1,0), self.mat_1.values[2])
        self.assertEqual(self.mat_1.get_value(1,1), self.mat_1.values[3])

        self.assertRaises(IndexError, self.mat_1.get_value, 2, 0)
        self.assertRaises(IndexError, self.mat_1.get_value, 0, 2)
        self.assertRaises(IndexError, self.mat_1.get_value, -1, 0)
        self.assertRaises(IndexError, self.mat_1.get_value, 0, -1)
        self.assertRaises(IndexError, self.mat_1.get_value, 2, 2)

    def test_add(self):
        self.assertEqual(self.mat_1 + self.mat_2, self.mat_1p2)
        self.assertEqual(self.mat_1 + self.mat_3, self.mat_1p3)
        self.assertEqual(self.mat_1 + self.mat_4, self.mat_1p4)

        self.assertEqual(self.mat_2 + self.mat_1, self.mat_1p2)
        self.assertEqual(self.mat_3 + self.mat_1, self.mat_1p3)
        self.assertEqual(self.mat_4 + self.mat_1, self.mat_1p4)

    def test_neg(self):
        self.assertEqual(-self.mat_1, Matrix(2, 2, values=[-1, -2, -3, -4]))

    def test_muls(self):
        self.assertEqual(2 * self.mat_1, Matrix(2, 2, values=[2, 4, 6, 8]))
        self.assertEqual(-2 * self.mat_1, Matrix(2, 2, values=[-2, -4, -6, -8]))
        self.assertEqual(2 * self.mat_5, Matrix(2, 1, values=[0, 2]))
        self.assertEqual(-2 * self.mat_6, Matrix(1, 3, values=[0, 0, -2]))

    def test_mul(self):
        self.assertEqual(self.mat_5 * self.mat_6, Matrix(2, 3, values=[0,0,0,0,0,1]))
        self.assertEqual(self.mat_b * self.mat_c * self.mat_a,
            Matrix(4, 1, values=[-6, -33, -36, -6]))

    def test_transpose(self):
        self.assertEqual(self.mat_1.transpose(), Matrix(2, 2, values=[1, 3, 2, 4]))
        self.assertEqual(self.mat_6.transpose(), Matrix(3, 1, values=[0, 0, 1]))

    def test_inverse(self):
        self.assertEqual(self.mat_d * self.mat_d.inverse(),Matrix.unit(3))
        self.assertEqual(self.mat_1 * self.mat_1.inverse(),Matrix.unit(2))

        self.assertEqual(self.mat_d.inverse() * self.mat_d,Matrix.unit(3))
        self.assertEqual(self.mat_1.inverse() * self.mat_1,Matrix.unit(2))
        self.assertEqual(self.mat_e.inverse() * self.mat_e,Matrix.unit(5))

    def test_unit(self):
        self.assertEqual(Matrix.unit(1),
            Matrix(1, 1, values=[1]))
        self.assertEqual(Matrix.unit(2),
            Matrix(2, 2, values=[1, 0, 0, 1]))
        self.assertEqual(Matrix.unit(3),
            Matrix(3, 3, values=[1, 0, 0, 0, 1, 0, 0, 0, 1]))

def TestSLE(unittest.TestCase):
    def setUp(self):
        self.sle_1 = SLE()


if __name__ == '__main__':
    unittest.main()
