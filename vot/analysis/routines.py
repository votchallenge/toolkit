from typing import List, Tuple

import numpy as np

from vot.dataset import Sequence
from vot.region.utils import calculate_overlaps
from vot.region import Region, Special
from vot.analysis import is_special

def compute_accuracy(trajectory: List[Region], sequence: Sequence, burnin: int = 10, 
    ignore_unknown: bool = True, bounded: bool = True) -> float:

    overlaps = np.array(calculate_overlaps(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    mask = np.ones(len(overlaps))

    for i, region in enumerate(trajectory):
        if is_special(region, Special.UNKNOWN) and ignore_unknown:
            mask[i] = False
        elif is_special(region, Special.INITIALIZATION):
            for j in range(i, i + burnin):
                mask[j] = False
        elif is_special(region, Special.FAILURE):
            mask[i] = False

    return np.mean(overlaps[mask]), np.sum(mask)

def count_failures(trajectory: List[Region]) -> Tuple[int, int]:
    return len([region for region in trajectory if is_special(region, Special.FAILURE)]), len(trajectory)

