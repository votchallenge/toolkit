import os
import json
import glob
import yaml
from typing import List

from vot.experiment import Experiment
from vot.experiment.transformer import Transformer
from vot.utilities import import_class
from vot.analysis import Analysis

class Stack(object):

    def __init__(self, workspace: "Workspace", metadata: dict):
        self._workspace = workspace

        self._title = metadata["title"]
        self._dataset = metadata.get("dataset", None)
        self._deprecated = metadata.get("deprecated", False)
        self._experiments = dict()
        self._analyses = dict()
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
                    transformers.append(transformer_class(workspace.cache, **transformer_metadata))

            analyses = []
            if "measures" in experiment_metadata:
                analyses_metadata = experiment_metadata["measures"]
                del experiment_metadata["measures"]

                for analysis_metadata in analyses_metadata:
                    analysis_class = import_class(analysis_metadata["type"], hints=["vot.analysis.measures"])
                    assert issubclass(analysis_class, Analysis)
                    del analysis_metadata["type"]
                    analyses.append(analysis_class(**analysis_metadata))
            experiment = experiment_class(_identifier=identifier, _storage=workspace._storage,
                    _transformers=transformers, **experiment_metadata)
            self._experiments[identifier] = experiment
            self._analyses[experiment] = [analysis for analysis in analyses if analysis.compatible(experiment)]
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
        return self._experiments.values()
        
    def analyses(self, experiment: Experiment) -> List["Analysis"]:
        return self._analyses[experiment]

    def transformers(self, experiment: Experiment) -> List["Transformer"]:
        return self._transformers[experiment]

    def __iter__(self):
        return iter(self._experiments.values())

    def __len__(self):
        return len(self._experiments)

    def __getitem__(self, identifier):
        return self._experiments[identifier]

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