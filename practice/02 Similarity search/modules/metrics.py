import numpy as np
import math


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

    # Вычисляем поэлементную разность
    diff = ts1 - ts2

    # Возводим разность в квадрат
    squared_diff = diff ** 2

    # Суммируем все квадраты
    sum_squared = np.sum(squared_diff)

    # Извлекаем квадратный корень
    ed_dist = np.sqrt(sum_squared)

    return ed_dist


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

    # Интерпретируем r как долю, если 0 <= r <= 1
    if 0 <= r <= 1:
        r_cells = math.floor(r * max(n, m))
    else:
        r_cells = int(r)

    # Минимальный размер окна, необходимый для существования
    # пути от (0, 0) до (n, m)
    r_cells = max(r_cells, abs(n - m))

    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0.0

    for i in range(1, n + 1):

        # Масштабируем положение диагонали для рядов разной длины.
        j_center = i * m / n

        j_start = max(1, math.floor(j_center - r_cells))
        j_end = min(m, math.ceil(j_center + r_cells))

        for j in range(j_start, j_end + 1):
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2

            d[i, j] = cost + min(
                d[i - 1, j],
                d[i, j - 1],
                d[i - 1, j - 1]
            )

    return d[n, m]
