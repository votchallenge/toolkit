Datasets
========

The easiest way to use the VOT toolkit is by using one of the integrated stacks that also provides a dataset. Everything is provided, and you just select a stack.
However, you can also use the toolkit with your own dataset. This document shows how to do this.

Background
----------

When loading an existing workspace, the toolkit will attempt to load anything that is available in the sequences subdirectory (by default this is `sequences`). The loading process is divided into two
steps, first an indexer will scan the directory for sequences and return the list of available sequences. Then, the sequence loader will load the sequence metadata into the appropriate structure.

This allows you some flexibility, you can organize your sequences in the format that the toolkit uses by default, or you can provide your own indexer and/or loader that will load the sequences from your custom format.
The toolkit comes with several loaders integrated, e.g. it can load sequences in formats for OTB, GoT10k, LaSOT, and TrackingNet. You can also provide your own loader, if you have a custom format and do not want to change it.

Default dataset format
----------------------

The default dataset format is a directory with subdirectories for each sequence, accompanied by a `list.txt` file that contains the list of sequences. 

Each sequence directory contains the following units: 

- Metadata (`sequence`): A file with sequence metadata in INI (key-value) format. The metadata also defines which channels are available for the sequence.
- Channels (usually `color`, `depth`, `ir`): Directories with images for each channel. The images are enumerated with a frame number and an extension that indicates the image format.
- Annotations (either `groundtruth.txt` or `groundtruth_<object>.json`): A file with ground truth annotations for the sequence. 
- Tags (denoted as `<name>.tag`): Per-frame tags that can be used to specify the binary state of each frame.
- Values (denoted as `<name>.value`): Per-frame values that can be used to specify numeric values for each frame.

The `list.txt` file contains the list of sequences in the dataset. Each line contains the name of a sequence directory.

Preparing a dataset
-------------------

To use a custom dataset, you have to prepare an empty workspace directory that already contains the `sequences` subdirectory that will be recognized by one of the integrated loaders. Then initialize the workspace with the `vot initialize` command with a desired stack of experiments.
Since the sequences are present, the command will not attempt to download them. If you would prefer to specify your own experiment stack, check out the tutorial on [creating a custom stack](stack.md).


Creating a custom loader
------------------------

As explained above, the toolkit uses an indexer and a loader to load sequences. Both are callable objects that accept a single string argument, the path to the directory. The indexer returns a list of sequence names, and the loader loads the Sequence object.
These callables are registered using the `class-registry <https://class-registry.readthedocs.io/>`_ package, and can be added to the registry using the setuptools entry points mechanism.

To create a custom indexer, you have to create a callable that returns a list of directories that contain sequences. An example of a custom indexer is shown below:

```python

def my_indexer(path):
    return ['sequence1', 'sequence2', 'sequence3']
```