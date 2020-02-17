import os
import json
import glob
from typing import List

from vot.experiment import Experiment
from vot.utilities import import_class

class Stack(object):

    def __init__(self, workspace: "Workspace", metadata: dict):
        from vot.analysis import PerformanceMeasure
        
        self._workspace = workspace

        self._title = metadata["title"]
        self._dataset = metadata.get("dataset", None)
        self._deprecated = metadata.get("deprecated", False)
        self._experiments = []
        self._measures = dict()

        for identifier, experiment_metadata in metadata["experiments"].items():
            experiment_class = import_class(experiment_metadata["type"])
            assert issubclass(experiment_class, Experiment)
            measures_metadata = experiment_metadata["measures"]
            del experiment_metadata["type"]
            del experiment_metadata["measures"]
            measures = []
            for measure_metadata in measures_metadata:
                measure_class = import_class(measure_metadata["type"])
                assert issubclass(measure_class, PerformanceMeasure)
                del measure_metadata["type"]
                measures.append(measure_class(**measure_metadata))
            experiment = experiment_class(identifier, workspace, **experiment_metadata)
            self._experiments.append(experiment)
            self._measures[experiment] = measures

    @property
    def title(self) -> str:
        return self._title

    @property
    def dataset(self) -> str:
        return self._dataset

    @property
    def deprecated(self) -> bool:
        return self._deprecated

    @property
    def workspace(self) -> "Workspace":
        return self._workspace

    @property
    def experiments(self) -> List[Experiment]:
        return self._experiments
        
    def measures(self, experiment: Experiment) -> List["PerformanceMeasure"]:
        return self._measures[experiment]

    def __iter__(self):
        return iter(self._experiments)

    def __len__(self):
        return len(self._experiments)

def resolve_stack(name, *directories):
    if os.path.isabs(name):
        return name if os.path.isfile(name) else None
    for directory in directories:
        full = os.path.join(directory, name)
        if os.path.isfile(full):
            return full
    full = os.path.join(os.path.dirname(__file__), name + ".yaml")
    if os.path.isfile(full):
        return full
    return None    