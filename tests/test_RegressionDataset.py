from gp.RegressionDataset import RegressionDataset
from check_assert import check_assert

import numpy as np  # type: ignore


def test_init():

    # give x, y
    x = np.array([[0],
                  [1]])
    y = np.array([[5],
                  [6]])
    f = lambda x: x[0]+5

    dataset = RegressionDataset(x=x, y=y)

    yield check_assert, dataset.x == x
    yield check_assert, dataset.y == y

    # give x, f
    dataset = RegressionDataset(x=x, f=f)

    yield check_assert, dataset.x == x
    yield check_assert, dataset.y == y
    yield check_assert, dataset.f == f


def test_linspace():

    # 1D test
    x = RegressionDataset.linspace(-1, 1, 4, 1)

    answer = np.array([[-1.],
                       [-1/3.],
                       [1/3.],
                       [1.]])

    x = np.round(x, 14)
    answer = np.round(answer, 14)

    yield check_assert, x.shape == (4, 1)
    yield check_assert, x == answer, 'RegressionDataset.linspace: 1D test (interval=[-1,1])'

    # 1D test in [-2, 0]
    x = RegressionDataset.linspace(-2, 0, 4, 1)

    answer = np.array([[-2.],
                       [-4/3.],
                       [-2/3.],
                       [0.]])

    x = np.round(x, 14)
    answer = np.round(answer, 14)

    yield check_assert, x.shape == (4, 1)
    yield check_assert, x == answer, 'RegressionDataset.linspace: 1D test (interval=[-2,0])'

    # 2D test
    x = RegressionDataset.linspace(-1, 1, 4, 2)

    answer = np.array([[-1., -1.],
                       [-1., 1.],
                       [1., -1.],
                       [1., 1.]])

    yield check_assert, x.shape == (4, 2)
    yield check_assert, x == answer, 'RegressionDataset.linspace: 2D test (interval=[-1,1])'


def test_urandspace():

    # 1D test
    x = RegressionDataset.urandspace(-1, 1, 4, 1)
    yield check_assert, x.shape == (4, 1)
    yield check_assert, (-1 <= x).all() and (x <= 1).all(), 'RegressionDataset.urandspace: 1D test (interval=[-1,1])'

    # 1D test in [-2, 0]
    x = RegressionDataset.urandspace(-2, 0, 4, 1)
    yield check_assert, x.shape == (4, 1)
    yield check_assert, (-2 <= x).all() and (x <= 0).all(), 'RegressionDataset.urandspace: 1D test (interval=[-2,0])'

    # 2D test
    x = RegressionDataset.urandspace(-1, 1, 4, 2)
    yield check_assert, x.shape == (4, 2)
    yield check_assert, (-1 <= x).all() and (x <= 1).all(), 'RegressionDataset.urandspace: 2D test (interval=[-1,1])'


def test_get_y():

    # 1D test
    x = np.array([[-1.0],
                  [-0.5],
                  [0.0],
                  [0.5],
                  [1.0]])

    f = lambda x: 2*x[0]

    y = RegressionDataset.get_y(x, f)

    answer = np.array([[-2.0],
                       [-1.0],
                       [0.0],
                       [1.0],
                       [2.0]])

    yield check_assert, y == answer, 'RegressionDataset.get_y: 1D test'
    yield check_assert, y.shape == (5, 1), 'y must be 2D'

    # with const
    f = lambda x, c: c[0]+c[1]*x[0]
    c = [1, 2]
    y = RegressionDataset.get_y(x=x, f=f, c=c)
    answer = np.array([[-1.0],
                       [0.0],
                       [1.0],
                       [2.0],
                       [3.0]])

    yield check_assert, y == answer, 'RegressionDataset.get_y: 1D test (constants)'
    yield check_assert, y.shape == (5, 1), 'y must be 2D'

    # 2D test
    x = np.array([[-1.0, -1.0],
                  [-0.5, -0.5],
                  [0.0, 0.0],
                  [0.5, 0.5],
                  [1.0, 1.0]])

    f = lambda x: 2*x[0]*x[1]

    y = RegressionDataset.get_y(x, f)

    answer = np.array([[2.0],
                       [0.5],
                       [0.0],
                       [0.5],
                       [2.0]])

    yield check_assert, y == answer, 'RegressionDataset.get_y: 2D test'
    yield check_assert, y.shape == (5, 1), 'y must be 2D'


def test_get_RMSE():

    x = np.array([[-1.],
                  [0.],
                  [1.]])

    y = np.array([[-2.],
                  [0.],
                  [2.]])

    d = RegressionDataset(x, y)

    function_list = [lambda x: 2*x[0],
                     lambda x: x[0],
                     lambda x: x[0]**2]

    answer_list = [0., np.sqrt(2/3), np.sqrt(10/3)]

    for f, ans in zip(function_list, answer_list):
        error = d.get_RMSE(f)

        yield check_assert, error == ans


def test_get_NRMSE():

    x = np.array([[-1.],
                  [0.],
                  [1.]])

    y = np.array([[-2.],
                  [0.],
                  [2.]])

    d = RegressionDataset(x, y)

    function_list = [lambda x: 2*x[0],
                     lambda x: x[0],
                     lambda x: x[0]**2]

    answer_list = [0., np.sqrt(2/3)/4, np.sqrt(10/3)/4]

    for f, ans in zip(function_list, answer_list):
        error = d.get_NRMSE(f)
        yield check_assert, error == ans

    # do again but change to constant function
    y = np.array([[0.],
                  [0.],
                  [0.]])
    d = RegressionDataset(x, y)

    answer_list = [np.sqrt(8/3), np.sqrt(2/3), np.sqrt(2/3)]

    for f, ans in zip(function_list, answer_list):
        error = d.get_NRMSE(f)
        yield check_assert, error == ans


def test_get_dataset():

    x = np.array([[0],
                  [1],
                  [2]])

    f = lambda x: x[0]**2

    dataset = RegressionDataset(x=x, f=f).get_dataset()

    answer = np.array([[0, 0],
                       [1, 1],
                       [2, 4]])

    yield check_assert, answer == dataset


def test_is_valid():

    # check: can inform x is not 2D
    x = np.linspace(-1, 1, 20)

    try:
        RegressionDataset.is_valid(x, y=None)
        yield check_assert, False, 'x is not 2D, valid did not detect it.'

    except AssertionError as error_message:
        print('FOUND THE ASSERTION', repr(error_message), error_message == 'Expected x to be 2D.')
        yield check_assert, str(error_message) == 'Expected x to be 2D.'

    # check: can inform y is not 2D
    x = x[:, None]
    y = np.linspace(-2, 2, 19)

    try:
        RegressionDataset.is_valid(x, y)
        yield check_assert, False, 'y is not 2D, valid did not detect it.'

    except AssertionError as error_message:
        yield check_assert, str(error_message) == 'Expected y to be 2D.'

    # check: x,y not the same shape
    y = y[:, None]

    try:
        RegressionDataset.is_valid(x, y)
        yield check_assert, False, 'y is not 2D, valid did not detect it.'

    except AssertionError as error_message:
        yield check_assert, str(error_message) == 'Expected same number of observations (rows) in x and y.'


def test_get_signed_error():

    dataset = RegressionDataset(x=[[0], [0.5], [1]],
                                f=lambda x: 3*x[0])
    answer_list = [[1., 1., 1.], [0., 0.5, 1.]]
    f_hat_list = [lambda x: 3*x[0]-1, lambda x: 2*x[0]]
    for answer, f_hat in zip(answer_list, f_hat_list):
        signed_error = dataset.get_signed_error(f_hat)
        yield check_assert, signed_error == answer


def test___len__():

    x = RegressionDataset.linspace(-1, 1, 20)
    f = lambda x: x[0]*3 + x[0]**3
    rd = RegressionDataset(x=x, f=f)
    yield check_assert, 20 == len(rd)

    x = RegressionDataset.linspace(-1, 1, 83)
    y = RegressionDataset.linspace(3, 8, 83)
    rd = RegressionDataset(x=x, y=y)
    yield check_assert, 83 == len(rd)


def test___eq__():
    x = RegressionDataset.linspace(-1, 1, 20)
    y = RegressionDataset.linspace(3, 8, 20)
    rd1 = RegressionDataset(x=x, y=y)
    rd2 = RegressionDataset(x=x, y=y)
    yield check_assert, rd1 == rd2

    rd2.y[0, 0] = 3.01
    yield check_assert, rd1 != rd2
