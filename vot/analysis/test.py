import vot 
from vot.tracker import Results, Trajectory
from vot.region.utils import calculate_overlap

# read ground truth:
gt_result = Results('/home/ondrej/Work/tracking/VOT/gt/bag')
gt_trajectory = Trajectory.read(gt_result, 'groundtruth')

# read result of tracker: 
result = Results('/home/ondrej/Work/tracking/VOT/dummy_results/KCF/baseline/bag')
trajectory = Trajectory.read(result, 'bag_001')

# compute overlap: 

# TODO: check gt_trajectory and trajectory are of same length

for r_gt, r in zip(gt_trajectory._regions, trajectory._regions):
    o = calculate_overlap(r_gt, r)
    print(o) 

    




