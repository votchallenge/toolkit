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

Step 6: Customize reports in `config.yaml`
------------------------------------------

Workspace report behavior is controlled by the `report` section in `config.yaml`.

The available configuration blocks are:

* `report.style` - plot/axes/figure callables,
* `report.sort` - tracker ordering before report generation,
* `report.index` - list of report generators to include.

Minimal custom index (equivalent to default overview):

.. code-block:: yaml

	report:
	  index:
	    - type: table
	    - type: plots

You can also include additional built-in report generators:

.. code-block:: yaml

	report:
	  index:
	    - type: table
	    - type: plots
	    - type: vot.report.common.SequenceOverlapPlots
	      ignore_masks: _ignore

For a description of each element see :ref:`report-elements` below.

Sort trackers by a selected analysis output before plots/tables are produced:

.. code-block:: yaml

	report:
	  sort:
	    experiment: baseline
	    analysis: <analysis-name>
	    result: 0

Notes:

* `experiment` must match experiment identifier from the stack (for example `baseline`).
* `analysis` must match `Analysis.name` (not necessarily class name).
* `result` is a zero-based index into analysis outputs returned by `describe()`.

Define local style hooks in your workspace (for example `report_hooks.py`):

.. code-block:: python

	from matplotlib.axes import Axes
	from vot.report import DefaultStyle

	class BasicStyle(DefaultStyle):
	    def line_style(self, opacity=1):
	        style = super().line_style(opacity=opacity)
	        style["linewidth"] = 2
	        return style

	    def point_style(self):
	        style = super().point_style()
	        style["s"] = 40
	        return style

	def basic_plot_style(index):
	    return BasicStyle(index)

	def basic_axes(figure, rect=None, _=None):
	    axes = Axes(figure, rect or [0, 0, 1, 1])
	    axes.grid(visible=True, color=[0.85, 0.85, 0.85], linestyle="--", linewidth=1)
	    figure.add_axes(axes)
	    return axes

Reference these hooks from `config.yaml`:

.. code-block:: yaml

	report:
	  style:
	    plots: report_hooks.basic_plot_style
	    axes: report_hooks.basic_axes

Explicit tracker-ordering example in `config.yaml`:

.. code-block:: yaml

	report:
	  sort:
	    experiment: baseline
	    analysis: Quality
	    result: 0

The `analysis` value above must match an analysis name that exists in your stack. If your stack defines a different name, replace `Quality` accordingly.

After editing `config.yaml`, regenerate the report:

.. code-block:: bash

	vot report --workspace ./workspace --format html tracker_a tracker_b

.. _report-elements:

Available report elements
--------------------------

All report elements are specified under ``report.index`` in ``config.yaml`` using the
``type`` key. Two short aliases are built in; all others must use the full Python class
path.

**table** (``vot.report.common.StackAnalysesTable``)
  Produces an overview table of all per-tracker scalar measures computed by the
  analyses defined in the stack. Each row is a tracker; each column group corresponds
  to one experiment/analysis combination. Columns are ranked so that the best result
  in each column is highlighted. This element has no parameters.

  .. code-block:: yaml

  	- type: table

**plots** (``vot.report.common.StackAnalysesPlots``)
  Produces one plot per analysis output (scatter plots for 2-D point results, line
  plots for curve/plot results) for every experiment in the stack. All trackers are
  drawn in reverse score order so that the best-performing tracker is rendered on top.
  This element has no parameters.

  .. code-block:: yaml

  	- type: plots

**SequenceOverlapPlots** (``vot.report.common.SequenceOverlapPlots``)
  Produces one overlap-over-time line plot per sequence for every experiment. Each
  line represents one tracker run. This element is only compatible with multi-run
  experiments (``MultiRunExperiment``); it is silently skipped for other experiment
  types.

  Parameters:

  * ``ignore_masks`` *(string, default ``_ignore``)* — object identifier used to
    fetch per-frame ignore masks from the dataset (regions marked with this label are
    excluded from the overlap calculation).

  .. code-block:: yaml

  	- type: vot.report.common.SequenceOverlapPlots
  	  ignore_masks: _ignore

**PreviewVideos** (``vot.report.video.PreviewVideos``)
	Produces rendered sequence preview videos with tracker trajectories overlaid on
	frames. This element is only compatible with multi-run experiments
	(``MultiRunExperiment``); it is silently skipped for other experiment types.

	Parameters:

	* ``groundtruth`` *(boolean, default ``false``)* - include ground-truth regions
		in the preview.
	* ``separate`` *(boolean, default ``false``)* - request separate videos per
		tracker. In current toolkit code, prefer the default combined mode.

	.. code-block:: yaml

		- type: vot.report.video.PreviewVideos
		  groundtruth: true
		  separate: false

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