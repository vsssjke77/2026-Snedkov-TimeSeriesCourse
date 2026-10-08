import numpy as np

from modules.utils import *


def top_k_discords(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k discords based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k: number of discords

    Returns
    --------
    discords: top-k discords (indices, distances to its nearest neighbor and the nearest neighbors indices)
    """
 
    discords_idx = []
    discords_dist = []
    discords_nn_idx = []

    mp = matrix_profile['mp'].copy()
    mpi = matrix_profile['mpi'].copy()
    m = matrix_profile['m']
    excl_zone = matrix_profile['excl_zone']

    for _ in range(top_k):
        # Находим индекс с максимальным значением матричного профиля
        idx = np.argmax(mp)

        if mp[idx] == -np.inf or mp[idx] == np.inf:
            break

        # Добавляем найденный диссонанс
        discords_idx.append(idx)
        discords_dist.append(mp[idx])
        discords_nn_idx.append(mpi[idx])

        # Применяем exclusion zone: массив → индекс → зона → значение
        mp = apply_exclusion_zone(mp, idx, excl_zone, -np.inf)

    return {
        'indices': discords_idx,
        'distances': discords_dist,
        'nn_indices': discords_nn_idx
    }
