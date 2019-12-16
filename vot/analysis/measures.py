import os
import vot
from vot.tracker import Results, Trajectory, Tracker
from vot.dataset import Sequence, VOTDataset, VOTSequence
from vot.region.utils import calculate_overlap, calculate_overlaps
from vot.region import Region, RegionType, Special
from vot.dataset import VOTDataset, VOTSequence, download_vot_dataset
from vot.experiment import SupervisedExperiment
from typing import List
import numpy as np

def is_region_special_code(r: Region, code: int):
    return r.type() == RegionType.SPECIAL and r.code() == code

# Re: Accuracy measure. Compute average overlap but ignore 10 frames after re-init.
# TODO: check what it means 10 frames after init (incl/excl init)
def compute_accuracy(regions_of_trajectory: List[Region], regions_gt: List[Region], frames_ignore_after_init=10) -> float:
    '''
    average overlap over usable frames (= those which are more than 10 frames from (re-)init)
    if there are no usable frames, returns -1.0
    '''
    # 1. compute overlaps:
    overlaps = calculate_overlaps(regions_of_trajectory, regions_gt)
    N = len(regions_of_trajectory)
    overlaps_for_average = [-1] * N
    idx_last_init = 0
    debug_test_ignored_frames = [0] * N
    for i, o, is_FAIL, is_INIT, is_UNKNOWN in zip(range(N),
                      overlaps,
                      [is_region_special_code(r, Special.FAILURE) for r in regions_of_trajectory],
                      [is_region_special_code(r, Special.INITIALIZATION) for r in regions_of_trajectory],
                      [is_region_special_code(r, Special.UNKNOWN) for r in regions_of_trajectory]):
        if is_INIT:
            idx_last_init = i
        if i <= idx_last_init + frames_ignore_after_init:
            debug_test_ignored_frames[i] = 1
        elif is_FAIL or is_UNKNOWN:
            pass
        else:
            overlaps_for_average[i] = o

    used_overlaps = [x for x in overlaps_for_average if x >= 0]
    average_overlap = np.array(used_overlaps).mean() if len(used_overlaps)>0 else -1.0

    return average_overlap, overlaps_for_average, debug_test_ignored_frames

# Re: Robustness measure. Count number of failures.
def count_failures(regions_of_trajectory: List[Region]) -> int:
        N = 0
        for r in regions_of_trajectory:
            if is_region_special_code(r, Special.FAILURE):
                N += 1
        return N


if __name__ == '__main__':
    #download_vot_dataset('vot2013') # this did not work for me but download was done using matlab implementation

    vot_dataset_directory = '/home/ondrej/Work/tracking/vot-w/sequences' #VOT2013
    results_directory = '/home/ondrej/Work/tracking/vot-w/results'
    experiment_type = 'baseline'
    tracker = Tracker('myNCC', 'dummy_command')
    experiment=SupervisedExperiment('SupExper_test')

    dataset = VOTDataset(vot_dataset_directory)
    sequence_names = dataset.list()


    for sequence_name in sequence_names:
        base = os.path.join(vot_dataset_directory, sequence_name)
        s = VOTSequence(base, name = sequence_name, dataset = dataset)
        results = Results(os.path.join(results_directory, tracker.identifier, experiment_type, s.name))
        complete, files = experiment.scan(tracker, s, results) # this is just my exercise, not further used
        # for now, take just 1st repetition:
        trajectory_file = s.name + '_001'

        trajectory = Trajectory.read(results, trajectory_file)
        # trajectory_length = # do we want to add this attribute to the Trajectory class?

        regions_gt = [s.groundtruth(i) for i in range(s.length)]
        regions_tr = [trajectory.region(i) for i in range(s.length)]
        print(count_failures(regions_tr))
        average_overlap, overlaps_for_average, debug_test_ignored_frames = compute_accuracy(regions_tr, regions_gt)
        print(average_overlap)

# read ground truth:
#gt_result = Results('/home/ondrej/Work/tracking/VOT/gt/bag')
#gt_trajectory = Trajectory.read(gt_result, 'groundtruth')

# read result of tracker: 
#result = Results('/home/ondrej/Work/tracking/VOT/dummy_results/KCF/baseline/bag')
#trajectory = Trajectory.read(result, 'bag_001')

# compute overlap: 

    




