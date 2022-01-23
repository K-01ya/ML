import numpy as np
from scipy.special import expit
from scipy.special import logsumexp
from scipy.sparse import csr_matrix


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """

    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.
    Оракул должен поддерживать l2 регуляризацию.
    """

    def __init__(self, l2_coef):
        """
        Задание параметров оракула.
        l2_coef - коэффициент l2 регуляризации
        """
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w - одномерный numpy array
        """
        if type(X) is np.ndarray:
            L = np.sum(np.logaddexp(np.zeros(y.shape[0]), -y * np.transpose(np.dot(X, np.transpose(w)))))
        else:
            L = np.sum(np.logaddexp(np.zeros(y.shape[0]), -y * X.dot(w)))
        return L / y.shape[0] + self.l2_coef / 2 * np.sum(w ** 2)

    def grad(self, X, y, w):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w - одномерный numpy array
        """

        if type(X) is np.ndarray:
            L_grad = np.dot((np.ones(y.shape[0]) - expit(y * np.transpose(np.dot(X, np.transpose(w))))),
                            -X * y.reshape((y.shape[0], 1)))
        else:
            L_grad = -X.transpose().dot(y * (expit(-y * X.dot(w))))
        return L_grad / y.shape[0] + self.l2_coef * w


if __name__ == '__main__':
    A = csr_matrix([[1, 2], [0, 0], [4, 0]])
    v = np.array([1, 0, -1])
    w = np.array([1, 2])
    bl = BinaryLogistic(1)
    # print(A.shape, v.shape, w.shape)
    print(bl.grad(A, v, w), bl.func(A, v, w))
