"""Stacks are collections of experiments that are grouped together for convenience. Stacks are used to organize experiments and to run them in 
batch mode.
"""
import os
from typing import List, Mapping

import yaml

from attributee import Attributee, String, Boolean, Map, Object
from attributee.io import Serializable

from vot.experiment import Experiment, experiment_registry


def experiment_resolver(typename, context, **kwargs):
    """Resolves experiment objects from stack definitions. This function is used by the stack module to resolve experiment objects from stack
    definitions. It is not intended to be used directly.

    Args:
        typename (str): Name of the experiment class
        context (Attributee): Context of the experiment
        kwargs (dict): Additional arguments

    Returns:
        Experiment: Experiment object
    """

    from vot.utilities import import_class

    identifier = context.key
    storage = None

    if getattr(context.parent, "workspace", None) is not None:
        storage = context.parent.workspace.storage

    if typename in experiment_registry:
        experiment = experiment_registry.get(typename, _identifier=identifier, _storage=storage, **kwargs)
        assert isinstance(experiment, Experiment)
        return experiment
    else:
        experiment_class = import_class(typename)
        assert issubclass(experiment_class, Experiment)
        return experiment_class(_identifier=identifier, _storage=storage, **kwargs)

class Stack(Attributee, Serializable):
    """Stack class represents a collection of experiments. Stacks are used to organize experiments and to run them in batch mode.
    """

    title = String(default="Stack")
    dataset = String(default=None)
    url = String(default="")
    deprecated = Boolean(default=False)
    experiments = Map(Object(experiment_resolver))

    @property
    def name(self):
        """Returns the name of the stack."""
        return getattr(self, "_name", None)

    def __iter__(self):
        """Iterates over experiments in the stack."""
        return iter(self.experiments.values())

    def __len__(self):
        """Returns the number of experiments in the stack."""
        return len(self.experiments)

    def __getitem__(self, identifier):
        """Returns the experiment with the given identifier.

        Args:
            identifier (str): Identifier of the experiment
        
        Returns:
            Experiment: Experiment object

        """
        return self.experiments[identifier]
    

def resolve_stack(name: str, *directories: List[str]) -> str:
    """Searches for stack file in the given directories and returns its absolute path. If given an absolute path as input
    it simply returns it.

    Args:
        name (str): Name of the stack
        directories (List[str]): Directories that will be used

    Returns:
        str: Absolute path to stack file
    """
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

def list_integrated_stacks() -> Mapping[str, str]:
    """List stacks that come with the toolkit

    Returns:
        Map[str, str]: A mapping of stack ids and stack title pairs
    """

    from pathlib import Path

    stacks = {}
    root = Path(os.path.join(os.path.dirname(__file__)))

    for stack_path in root.rglob("*.yaml"):
        with open(stack_path, 'r') as fp:
            stack_metadata = yaml.load(fp, Loader=yaml.BaseLoader)
        if stack_metadata is None:
            continue
        key = str(stack_path.relative_to(root).with_name(os.path.splitext(stack_path.name)[0]))
        stacks[key] = stack_metadata.get("title", "")

    return stacks