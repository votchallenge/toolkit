from typing import List, Tuple
import math

import numpy as np

from vot.dataset import Sequence
from vot.region.utils import calculate_overlaps
from vot.region import Region, Special, RegionType
from vot.analysis import is_special

def compute_accuracy(trajectory: List[Region], sequence: Sequence, burnin: int = 10, 
    ignore_unknown: bool = True, bounded: bool = True) -> float:

    overlaps = np.array(calculate_overlaps(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    mask = np.ones(len(overlaps), dtype=bool)

    for i, region in enumerate(trajectory):
        if is_special(region, Special.UNKNOWN) and ignore_unknown:
            mask[i] = False
        elif is_special(region, Special.INITIALIZATION):
            for j in range(i, min(len(trajectory), i + burnin)):
                mask[j] = False
        elif is_special(region, Special.FAILURE):
            mask[i] = False
    
    if any(mask):
        return np.mean(overlaps[mask]), np.sum(mask)
    else:
        return 0, 0

def count_failures(trajectory: List[Region]) -> Tuple[int, int]:
    return len([region for region in trajectory if is_special(region, Special.FAILURE)]), len(trajectory)

def locate_failures_inits(trajectory: List[Region]) -> Tuple[int, int]:
    return [i for i, region in enumerate(trajectory) if is_special(region, Special.FAILURE)], \
            [i for i, region in enumerate(trajectory) if is_special(region, Special.INITIALIZATION)]

def compute_eao(overlaps: List, weights: List[float], success: List[bool], bound_low: int, bound_high: int):
    max_length = max([len(el) for el in overlaps])
    total_runs = len(overlaps)
    
    overlaps_array = np.zeros((total_runs, max_length), dtype=np.float32)
    mask_array = np.zeros((total_runs, max_length), dtype=np.float32)  # mask out frames which are not considered in EAO calculation
    weights_vector = np.reshape(np.array(weights, dtype=np.float32), (len(weights), 1))  # weight of each run

    for i, (o, success) in enumerate(zip(overlaps, success)):
        overlaps_array[i, :len(o)] = np.array(o)
        if not success:
            # tracker has failed during this run - fill zeros until the end of the run
            mask_array[i, :] = 1
        else:
            # tracker has successfully tracked to the end - consider only this part of the sequence
            mask_array[i, :len(o)] = 1

    overlaps_array_sum = overlaps_array.copy()
    for j in range(1, overlaps_array_sum.shape[1]):
        overlaps_array_sum[:, j] = np.mean(overlaps_array[:, 1:j+1], axis=1)
    
    eao_curve = np.sum(weights_vector * overlaps_array_sum * mask_array, axis=0) / np.sum(mask_array * weights_vector, axis=0)
    
    return float(np.mean(eao_curve[bound_low:bound_high + 1])), eao_curve

def determine_thresholds(scores: List[float], resolution: int) -> List[float]:
    scores = [score for score in scores if not math.isnan(score)]
    scores = sorted(scores, reverse=True)

    if len(scores) > resolution - 2:
        delta = math.floor(len(scores) / (resolution - 2))
        idxs = np.round(np.linspace(delta, len(scores) - delta, num = resolution - 2)).astype(np.int)
        thresholds = [scores[idx] for idx in idxs]
    else:
        thresholds = scores

    thresholds.insert(0, math.inf)
    thresholds.insert(len(thresholds), -math.inf)

    return thresholds

def compute_tpr_curves(trajectory: List[Region], confidence: List[float], sequence: Sequence, thresholds: List[float], 
    ignore_unknown: bool = True, bounded: bool = True):
    
    overlaps = np.array(calculate_overlaps(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    confidence = np.array(confidence)

    n_visible = len([region for region in sequence.groundtruth() if region.type is not RegionType.SPECIAL])

    precision = len(thresholds) * [float(0)]
    recall = len(thresholds) * [float(0)]

    for i in range(len(thresholds)):

        subset = confidence >= thresholds[i]

        if np.sum(subset) == 0:
            precision[i] = 1
            recall[i] = 0
        else:
            precision[i] = np.mean(overlaps[subset])
            recall[i] = np.sum(overlaps[subset]) / n_visible

    return precision, recall