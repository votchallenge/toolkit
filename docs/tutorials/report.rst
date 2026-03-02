Report generation
=================

This tutorial shows how to generate reports from tracker results in a workspace.

Goal
----

At the end, you will be able to generate:

* an HTML report,
* a LaTeX report,
* plot/video exports.

Prerequisites
-------------

Before generating a report, make sure you have:

* an initialized workspace,
* at least one integrated tracker,
* evaluation results (run `vot evaluate`).

Step 1: Run evaluation
----------------------

Generate tracker outputs first:

.. code-block:: bash

	vot evaluate --workspace ./workspace mytracker

You can provide multiple trackers:

.. code-block:: bash

	vot evaluate --workspace ./workspace tracker_a tracker_b

Step 2: Generate a basic HTML report
------------------------------------

Create a default report:

.. code-block:: bash

	vot report --workspace ./workspace --format html mytracker

If `--name` is not provided, the toolkit creates a timestamp-based report name.

Step 3: Choose report format
----------------------------

The command supports three formats:

* `html` - interactive browser report,
* `latex` - LaTeX document output,
* `plots` - exported plot/video files only.

Examples:

.. code-block:: bash

	vot report --workspace ./workspace --format latex --name paper_report mytracker
	vot report --workspace ./workspace --format plots --name figures mytracker

Step 4: Filter report content
-----------------------------

You can restrict report generation to selected experiments and sequences.

.. code-block:: bash

	vot report --workspace ./workspace \
	  --experiments baseline,realtime \
	  --sequences bike1,bike2 \
	  --format html \
	  --name focused_report \
	  mytracker

This is useful for faster iteration when debugging integrations or preparing a concise comparison.

Step 5: Compare multiple trackers
---------------------------------

Generate one report with multiple trackers:

.. code-block:: bash

	vot report --workspace ./workspace --format html tracker_a tracker_b tracker_c

The toolkit sorts trackers according to report configuration and places all selected trackers in one output document.

Where outputs are stored
------------------------

Report outputs are stored in:

.. code-block:: text

	<workspace>/reports/<report-name>/

Typical files include:

* `report.html` for HTML format,
* `report.tex` for LaTeX format,
* `*.png`, `*.pdf`, `*.avi` exports for plots format.

Notes on report content
-----------------------

By default, report content is based on report configuration in the workspace.
If no custom report configuration is present, the toolkit generates a default overview (analysis table and analysis plots).

Troubleshooting
---------------

* **No trackers resolved**: verify tracker identifier in `trackers.ini` and command arguments.
* **No experiments/sequences selected**: check names used in `--experiments` and `--sequences`.
* **Empty report**: ensure evaluation has completed for the selected tracker(s).
* **Missing dependencies for rendering**: start with `--format html` (most robust default).