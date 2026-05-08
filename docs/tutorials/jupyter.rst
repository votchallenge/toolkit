Interactive analysis in Jupyter notebooks
=========================================

This tutorial covers the notebook-specific utilities in ``vot.utilities.notebook``.
They provide interactive widgets, progress feedback, and convenience wrappers
around the toolkit's core workspace, experiment, and analysis APIs — all designed
to work directly inside a Jupyter notebook kernel.

A self-contained demo notebook is available in the docs folder:

.. code-block:: text

	data/notebook_demo.ipynb

Goal
----

At the end, you will be able to:

* load a workspace and resolve trackers from a notebook,
* visualize sequence frames and ground truth,
* run experiments with inline progress bars,
* run stack analyses and optionally serialize results,
* overlay stored results in an interactive frame-by-frame widget,
* run a tracker live and step through frames interactively.

Prerequisites
-------------

Install the required notebook dependencies:

.. code-block:: bash

	pip install ipywidgets

Make sure you have:

* an initialized workspace,
* at least one tracker registered in the workspace.

Step 1: Import and load the workspace
--------------------------------------

.. code-block:: python

	from pathlib import Path
	from vot.workspace import Workspace
	import vot.utilities.notebook as vnb

	workspace = Workspace.load("/path/to/workspace")
	print("Experiments:", len(workspace.stack))
	print("Sequences:", len(workspace.dataset))

Step 2: Resolve trackers
-------------------------

Resolve trackers from the workspace registry. The code below falls back to all
trackers that already have results when no explicit list is given:

.. code-block:: python

	results_storage = workspace.storage.substorage("results")

	TRACKER_IDS = ["mytracker"]   # or [] to use all trackers with results

	if TRACKER_IDS:
		trackers = workspace.registry.resolve(
			*TRACKER_IDS,
			storage=results_storage,
			skip_unknown=False,
		)
	else:
		trackers = workspace.list_results(workspace.registry)

Select an experiment and a sequence to work with:

.. code-block:: python

	EXPERIMENT_ID = "baseline"   # must match an identifier in the stack
	SEQUENCE_NAME = "bag"        # must match a sequence name in the dataset

	experiment = next(e for e in workspace.stack if e.identifier == EXPERIMENT_ID)
	sequence   = next(s for s in workspace.dataset if s.name == SEQUENCE_NAME)

Step 3: Visualize a sequence frame — SequenceView
--------------------------------------------------

``SequenceView`` displays one frame at a time with ground truth drawn in green.
An optional overlay region (for example a tracker output) is drawn in red.

.. code-block:: python

	view = vnb.SequenceView(sequence)
	view.set_frame(0)           # choose frame index; optional region= argument
	view.show()                 # renders an ipywidgets widget inline

The widget updates in place: call ``view.set_frame(index, region=some_region)``
from subsequent cells to step through frames.

Step 4: Run an experiment
--------------------------

``run_experiment`` executes one experiment for the given trackers and sequences
and shows an inline progress bar:

.. code-block:: python

	vnb.run_experiment(
		experiment=experiment,
		sequences=[sequence],
		trackers=[trackers[0]],
		force=False,    # set True to overwrite existing results
		persist=True,   # continue even if a tracker raises an error
	)

Passing multiple sequences and trackers runs them all within the same call.

Step 5: Run stack analyses
---------------------------

``run_analysis`` computes all analyses defined in the stack for the given
trackers and sequences and returns the results in memory:

.. code-block:: python

	results = vnb.run_analysis(
		workspace=workspace,
		trackers=[trackers[0]],
		sequences=[sequence.name],          # optional: filter to specific sequences
		experiments=[experiment.identifier], # optional: filter to specific experiments
	)

To serialize the output to workspace storage at the same time:

.. code-block:: python

	vnb.run_analysis(
		workspace=workspace,
		trackers=[trackers[0]],
		output_format="json",   # "json" or "yaml"
		name="my_analysis",     # omit to use a timestamp-based name
	)

Serialized files are written to ``<workspace>/analysis/``.

Step 6: Visualize stored results
---------------------------------

``visualize_results`` opens an interactive frame-by-frame widget overlaying
stored tracker trajectories on the sequence. Ground truth is green;
trackers cycle through red, blue, orange, purple, and yellow:

.. code-block:: python

	vnb.visualize_results(
		experiment=experiment,
		sequence=sequence,
		trackers=[trackers[0]],   # pass multiple trackers to compare
	)

Use the **Restart**, **Next**, and **Run/Stop** buttons to control playback.

Step 7: Run a tracker live
---------------------------

``visualize_tracker`` starts the actual tracker process and displays output
frame-by-frame. This is useful for debugging integration or comparing tracker
behavior against stored results:

.. code-block:: python

	vnb.visualize_tracker(trackers[0], sequence)

.. warning::

	This call launches an external tracker process. Make sure the tracker is
	properly integrated before running it interactively (see the :doc:`integration`
	tutorial).

The widget exposes the same **Restart**, **Next**, and **Run/Stop** controls.

Environment check
-----------------

``vot.utilities.notebook`` guards every visualization function against accidental
calls outside a notebook kernel. You can also check explicitly:

.. code-block:: python

	import vot.utilities.notebook as vnb

	if not vnb.is_notebook():
		print("Not running inside a Jupyter kernel.")

All visualization functions raise ``ImportError`` when called outside a notebook
or when ``ipywidgets`` is not installed.

Troubleshooting
---------------

* **Widgets not displayed**: run ``jupyter nbextension enable --py widgetsnbextension``
  (classic Jupyter) or install the JupyterLab widgets extension.
* **Tracker process fails to start**: confirm that ``vot test <tracker-id>`` works
  from the command line before using ``visualize_tracker``.
* **No trackers resolved**: check that tracker identifiers in ``TRACKER_IDS`` match
  entries in ``trackers.ini`` or that results folders exist in
  ``<workspace>/results/``.
* **``run_analysis`` returns ``None``**: results for the requested
  sequences/experiments are missing — run the experiment first.