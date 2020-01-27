
from vot.stack import Stack
from vot.experiment import UnsupervisedExperiment, SupervisedExperiment, MultiStartExperiment

class VOTBasicTest(Stack):

    dataset = "test"

    def __init__(self):
        super().__init__(UnsupervisedExperiment("unsupervised", repetitions=1), SupervisedExperiment("supervised", 1, 10))

class VOTSegmentation(Stack):

    dataset = "segmentation"

    def __init__(self):
        super().__init__(MultiStartExperiment("default"))