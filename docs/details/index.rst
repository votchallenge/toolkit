Toolkit Details
===============

.. toctree::
   :maxdepth: 1

   dataset
   protocols
   stack
   workspace
   report

Key concepts that are used throughout the toolkit are:

* **Dataset** - a collection of sequences that is used for performance evaluation. A dataset is a collection of **sequences** and can be inported from `various formats <dataset.rst>`_.
* **Sequence** - a sequence of frames with correspoding ground truth annotations for one or more objects. A sequence is a collection of **frames**.
* **Tracker** - a tracker is an algorithm that takes frames from a sequence as input (one by one) and produces a set of **trajectories** as output. Each tracker can be implemented in a different way and can be integrated into the toolkit with one of the `protocols <protocols.rst>`_.
* **Experiment** - an experiment is a method that applies a tracker to a given sequence in a specific way.
* **Analysis** - an analysis is a set of **measures** that are used to evaluate the performance of a tracker (compare predicted trajectories to groundtruth).
* **Stack** - a stack is a collection of **experiments** and **analyses** that are performed on a given dataset, detailed information available `here <stack.rst>`_.
* **Workspace** - a workspace is a collection of experiments and analyses that are performed on a given dataset, detailed information available `here <workspace.rst>`_.
* **Report** - a report is a representation of a list of analyses for a given experiment stack.

