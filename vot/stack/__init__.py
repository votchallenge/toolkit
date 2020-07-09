import os
import json
import glob
import collections
from typing import List

import yaml

from vot.experiment import Experiment
from vot.experiment.transformer import Transformer
from vot.utilities import import_class
from vot.analysis import Analysis
from vot.utilities.attributes import Attributee, String, Boolean, Map, Object

def experiment_resolver(typename, context, **kwargs):
    experiment_class = import_class(typename)
    assert issubclass(experiment_class, Experiment)
    if "key" in context:
        identifier = context["key"]
    else:
        identifier = None

    storage = None
    if "parent" in context:
        if getattr(context["parent"], "workspace", None) is not None:
            storage = context["parent"].workspace.storage

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