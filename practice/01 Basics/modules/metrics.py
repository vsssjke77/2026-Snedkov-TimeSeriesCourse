import numpy as np


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """

    return np.sqrt(np.sum((ts1 - ts2) ** 2))


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """

    n = len(ts1)

    # Вычисляем средние
    mu1 = np.mean(ts1)
    mu2 = np.mean(ts2)

    # Вычисляем стандартные отклонения
    sigma1 = np.std(ts1)
    sigma2 = np.std(ts2)

    # Скалярное произведение
    dot_product = np.dot(ts1, ts2)

    # Вычисляем числитель (корреляцию)
    correlation = (dot_product - n * mu1 * mu2) / (n * sigma1 * sigma2)

    # Вычисляем нормализованное евклидово расстояние
    norm_ed_dist = np.sqrt(abs(2 * n * (1 - correlation)))

    return norm_ed_dist


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    n = len(ts1)
    m = len(ts2)

    # Создаем матрицу (n+1) x (m+1), заполняем бесконечностью
    dtw_matrix = np.full((n + 1, m + 1), np.inf)

    # Начальная точка - стоимость 0
    dtw_matrix[0, 0] = 0

    # Заполняем матрицу построчно
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Считаем разницу между текущими точками (возводим в квадрат)
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2

            # Берем минимальный путь из трех возможных
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],  # сдвиг в первом ряду
                dtw_matrix[i, j - 1],  # сдвиг во втором ряду
                dtw_matrix[i - 1, j - 1]  # совпадение
            )

    # Возвращаем корень из значения в правом нижнем углу
    return dtw_matrix[n, m]
