

from vot.stack import Stack
from vot.experiment import UnsupervisedExperiment, SupervisedExperiment

class VOT2013(Stack):

    deprecated = True
    dataset = "vot2013"

    def __init__(self):
        super().__init__(SupervisedExperiment("baseline", repetitions=15, skip_initialize=5))

class VOT2014(Stack):

    deprecated = True
    dataset = "vot2014"

    def __init__(self):
        super().__init__(SupervisedExperiment("baseline", repetitions=15, skip_initialize=5))

class VOT2015(Stack):

    dataset = "vot2015"

    def __init__(self):
        super().__init__(SupervisedExperiment("baseline", repetitions=15, skip_initialize=5))

class VOT2016(Stack):

    dataset = "vot2016"

    def __init__(self):
        super().__init__(SupervisedExperiment("baseline", repetitions=15, skip_initialize=5), \
             UnsupervisedExperiment("unsupervised", repetitions=1))

class VOT2017(Stack):

    dataset = "vot2017"

    def __init__(self):
        super().__init__(SupervisedExperiment("baseline", repetitions=15, skip_initialize=5), \
             UnsupervisedExperiment("unsupervised", repetitions=1))

class VOT2018(Stack):

    dataset = "vot2018"

    def __init__(self):
        super().__init__(SupervisedExperiment("baseline", repetitions=15, skip_initialize=5), \
             UnsupervisedExperiment("unsupervised", repetitions=1))

class VOT2019(Stack):

    dataset = "vot2019"

    def __init__(self):
        super().__init__(SupervisedExperiment("baseline", repetitions=15, skip_initialize=5), \
             UnsupervisedExperiment("unsupervised", repetitions=1))