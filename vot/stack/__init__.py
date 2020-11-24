import os
import json
import glob
import collections
from typing import List

import yaml

from attributee import Attributee, String, Boolean, Map, Object

from vot.experiment import Experiment, experiment_registry
from vot.experiment.transformer import Transformer
from vot.utilities import import_class
from vot.analysis import Analysis

def experiment_resolver(typename, context, **kwargs):

    if "key" in context:
        identifier = context["key"]
    else:
        identifier = None

    storage = None
    if "parent" in context:
        if getattr(context["parent"], "workspace", None) is not None:
            storage = context["parent"].workspace.storage

    if typename in experiment_registry:
        experiment = experiment_registry.get(typename, _identifier=identifier, _storage=storage, **kwargs)
        assert isinstance(experiment, Experiment)
    else:
        experiment_class = import_class(typename)
        assert issubclass(experiment_class, Experiment)
        return experiment_class(_identifier=identifier, _storage=storage, **kwargs)

class Stack(Attributee):

    title = String()
    dataset = String(default="")
    url = String(default="")
    deprecated = Boolean(default=False)
    experiments = Map(Object(experiment_resolver))

    def __init__(self, name: str, workspace: "Workspace", **kwargs):
        self._workspace = workspace
        self._name = name

        super().__init__(**kwargs)

    @property
    def workspace(self):
        return self._workspace

    @property
    def name(self):
        return self._name

    def __iter__(self):
        return iter(self.experiments.values())

    def __len__(self):
        return len(self.experiments)

    def __getitem__(self, identifier):
        return self.experiments[identifier]

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