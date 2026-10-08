import numpy as np

from modules.utils import *


def top_k_motifs(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k motifs based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k : number of motifs

    Returns
    --------
    motifs: top-k motifs (left and right indices and distances)
    """

    motifs_idx = []
    motifs_dist = []

    mp = matrix_profile['mp'].copy()
    mpi = matrix_profile['mpi'].copy()
    m = matrix_profile['m']
    excl_zone = matrix_profile['excl_zone']

    for _ in range(top_k):
        idx = np.argmin(mp)

        if mp[idx] == np.inf:
            break

        # Сортируем пару индексов, чтобы left < right (важно для plot_motifs)
        left_idx = min(idx, mpi[idx])
        right_idx = max(idx, mpi[idx])

        motifs_idx.append([left_idx, right_idx])
        motifs_dist.append(mp[idx])

        # Применяем exclusion zone к обоим индексам пары
        mp = apply_exclusion_zone(mp, left_idx, excl_zone, np.inf)
        mp = apply_exclusion_zone(mp, right_idx, excl_zone, np.inf)

    return {
        "indices": motifs_idx,
        "distances": motifs_dist
    }
