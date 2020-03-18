import os
import json
import glob
import yaml
from typing import List

from vot.experiment import Experiment
from vot.experiment.transformer import Transformer
from vot.utilities import import_class
from vot.analysis import PerformanceMeasure

class Stack(object):

    def __init__(self, workspace: "Workspace", metadata: dict):
        from vot.analysis import PerformanceMeasure
        
        self._workspace = workspace

        self._title = metadata["title"]
        self._dataset = metadata.get("dataset", None)
        self._deprecated = metadata.get("deprecated", False)
        self._experiments = []
        self._measures = dict()
        self._transformers = dict()

        for identifier, experiment_metadata in metadata["experiments"].items():
            experiment_class = import_class(experiment_metadata["type"], hints=["vot.experiment"])
            assert issubclass(experiment_class, Experiment)
            del experiment_metadata["type"]

            transformers = []
            if "transformers" in experiment_metadata:
                transformers_metadata = experiment_metadata["transformers"]
                del experiment_metadata["transformers"]

                for transformer_metadata in transformers_metadata:
                    transformer_class = import_class(transformer_metadata["type"], hints=["vot.experiment.transformer"])
                    assert issubclass(transformer_class, Transformer)
                    del transformer_metadata["type"]
                    transformers.append(transformer_class(workspace, **transformer_metadata))

            measures = []
            if "measures" in experiment_metadata:
                measures_metadata = experiment_metadata["measures"]
                del experiment_metadata["measures"]

                for measure_metadata in measures_metadata:
                    measure_class = import_class(measure_metadata["type"], hints=["vot.analysis.measures"])
                    assert issubclass(measure_class, PerformanceMeasure)
                    del measure_metadata["type"]
                    measures.append(measure_class(**measure_metadata))
            experiment = experiment_class(identifier, workspace, **experiment_metadata)
            self._experiments.append(experiment)
            self._measures[experiment] = measures
            self._transformers[experiment] = transformers

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

    def transformers(self, experiment: Experiment) -> List["Transformer"]:
        return self._transformers[experiment]

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

def list_integrated_stacks():
    stacks = {}
    for stack_file in glob.glob(os.path.join(os.path.dirname(__file__), "*.yaml")):
        with open(stack_file, 'r') as fp:
            stack_metadata = yaml.load(fp, Loader=yaml.BaseLoader)
        stacks[os.path.splitext(os.path.basename(stack_file))[0]] = stack_metadata.get("title", "")

    return stacks    