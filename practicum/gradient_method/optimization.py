import numpy as np
import time
from oracles import BinaryLogistic
from scipy.special import expit
from scipy.sparse import csr_matrix


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """
    def __init__(
            self, loss_function, step_alpha=0, step_beta=1,
            tolerance=1e-5, max_iter=1000,  **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        step_alpha - float, параметр выбора шага из текста задания
        step_beta- float, параметр выбора шага из текста задания
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход
        max_iter - максимальное число итераций
        **kwargs - аргументы, необходимые для инициализации
        """
        self.loss_function = loss_function
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.l2_coef = kwargs['l2_coef']
        self.w = None
        self.oracle = BinaryLogistic(self.l2_coef)

    def fit(self, X, y, w_0=None, trace=False, accuracy=False, X_test=None, y_test=None):
        """
        Обучение метода по выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w_0 - начальное приближение в методе
        trace - переменная типа bool
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)
        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        if w_0 is None:
            self.w = np.zeros(X.shape[1])
        else:
            self.w = w_0
        if trace:
            history = {'time': [], 'func': [], 'accuracy': []}
        previous_func_value = 0

        for k in range(1, self.max_iter + 1):
            learning_rate = self.step_alpha / (k ** self.step_beta)
            start = time.time()
            func_value = self.oracle.func(X, y, self.w)
            self.w -= learning_rate * self.oracle.grad(X, y, self.w)
            if trace:
                history['time'].append(time.time() - start)
                history['func'].append(func_value)
                if accuracy:
                    history['accuracy'].append(sum(np.sign(X_test.dot(self.w)) == y_test) / y_test.shape[0])
            if abs(func_value - previous_func_value) < self.tolerance:
                break
            previous_func_value = func_value

        if trace:
            return history

    def predict(self, X):
        """
        Получение меток ответов на выборке X
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        return: одномерный nu   mpy array с предсказаниями
        """

        return np.sign(X.dot(self.w))

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k
        """
        return expit(X.dot(self.w))

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        return: float
        """
        return self.oracle.func(X, y, self.w)

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        return: numpy array, размерность зависит от задачи
        """
        return self.oracle.grad(X, y, self.w)

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self.w


class SGDClassifier(GDClassifier):
    def __init__(self, loss_function, batch_size, step_alpha=1, step_beta=0, tolerance=1e-5, max_iter=1000,
                 random_seed=153, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        batch_size - размер подвыборки, по которой считается градиент
        step_alpha - float, параметр выбора шага из текста задания
        step_beta- float, параметр выбора шага из текста задания
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход
        max_iter - максимальное число итераций (эпох)
        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.
        **kwargs - аргументы, необходимые для инициализации
        """
        super().__init__(loss_function, step_alpha, step_beta, tolerance, max_iter, **kwargs)
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.l2_coef = kwargs['l2_coef']
        self.w = None
        self.oracle = BinaryLogistic(self.l2_coef)

    def fit(self, X, y, w_0=None, trace=False, log_freq=1., accuracy=False, X_test=None, y_test=None):
        """
        Обучение метода по выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w_0 - начальное приближение в методе
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}
        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления.
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.
        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        if w_0 is None:
            self.w = np.zeros(X.shape[1])
        else:
            self.w = w_0
        if trace:
            history = {'time': [], 'func': [], 'epoch_num': [], 'weights_diff': [], 'accuracy': []}
        previous_func_value = 0
        k = 1
        indices = np.arange(y.shape[0])
        processed = 0
        previous_processed = 0
        previous_weights = 0

        while (processed // y.shape[0]) < self.max_iter:
            np.random.shuffle(indices)
            learning_rate = self.step_alpha / (k ** self.step_beta)
            start = time.time()

            iteration = 0
            while (processed - previous_processed) / y.shape[0] < log_freq:
                iteration_indices = indices[iteration * self.batch_size:(iteration + 1) * self.batch_size + 1]
                self.w -= learning_rate * self.oracle.grad(X[iteration_indices], y[iteration_indices], self.w)
                iteration += 1
                processed += self.batch_size

            func_value = self.oracle.func(X, y, self.w)
            if trace:
                history['time'].append(time.time() - start)
                history['func'].append(func_value)
                history['weights_diff'].append(np.linalg.norm(self.w - previous_weights) ** 2)
                history['epoch_num'].append(processed / y.shape[0])
                if accuracy:
                    history['accuracy'].append(sum(np.sign(X_test.dot(self.w)) == y_test) / y_test.shape[0])
            if abs(func_value - previous_func_value) < self.tolerance:
                break
            previous_weights = self.w
            previous_processed = processed
            previous_func_value = func_value
            k += 1

        if trace:
            return history


if __name__ == '__main__':
    np.random.seed(19)
    clf = GDClassifier(loss_function='binary_logistic', step_alpha=1, step_beta=0, tolerance=1e-4, max_iter=5,
                        l2_coef=0.1, batch_size=1)
    l, d = 1000, 10
    X = np.random.random((l, d))
    y = np.random.randint(0, 2, l) * 2 - 1
    w = np.random.random(d)
    history = clf.fit(X, y, w_0=np.zeros(d), trace=True)
    # print(' '.join([str(x) for x in history['func']]))
    print(clf.predict(csr_matrix(X)))
    print(history['accuracy'])
