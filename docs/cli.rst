Using the CLI
=============

The toolkit command-line interface is available as `vot` (or `python -m vot`).
It is the main entry point for workspace setup, tracker testing, evaluation, analysis, report generation, and result packaging.

General usage
-------------

.. code-block:: bash

	vot [global-options] <command> [command-options]

Global options:

* `--debug`, `-d` - enable debug mode and verbose logs,
* `--registry <path>` - add tracker registry path (directory or registry file).

Get command help:

.. code-block:: bash

	vot --help
	vot <command> --help

Commands overview
-----------------

* `initialize` (alias: `configure`) - initialize workspace,
* `test` - run tracker integration test on synthetic (or provided) sequence,
* `evaluate` (alias: `run`) - execute experiments for one or more trackers,
* `analysis` (aliases: `analyse`, `analyze`) - compute analysis outputs,
* `report` (alias: `document`) - generate report documents,
* `pack` - package tracker results for submission.

Initialize a workspace
----------------------

Initialize a workspace and configure stack/dataset.

.. code-block:: bash

	vot initialize [--workspace <path>] [--nodownload] [<stack>]

Arguments and options:

* `<stack>` - optional stack identifier,
* `--workspace <path>` - workspace directory (default: current directory),
* `--nodownload` - skip dataset download from stack metadata.

Examples:

.. code-block:: bash

	vot initialize vots2025/main --workspace ./workspace
	vot configure --workspace ./workspace --nodownload

Test a tracker
--------------

Run tracker integration test.

.. code-block:: bash

	vot test [--visualize] [--sequence <path>] [--ignore <ids>] <tracker>

Arguments and options:

* `<tracker>` - tracker identifier from registry,
* `--visualize`, `-g` - show visualized test run,
* `--sequence`, `-s <path>` - use custom sequence instead of synthetic sequence,
* `--ignore <id1,id2,...>` - ignore selected object identifiers.

Example:

.. code-block:: bash

	vot test -g mytracker

Run evaluation
--------------

Run experiment stack for one or more trackers.

.. code-block:: bash

	vot evaluate [--workspace <path>] [--force] [--persist] [--experiments <ids>] <tracker> [<tracker> ...]

Arguments and options:

* `<tracker> [<tracker> ...]` - one or more tracker identifiers,
* `--workspace <path>` - workspace directory,
* `--force`, `-f` - force full rerun,
* `--persist`, `-p` - continue even when errors are encountered,
* `--experiments <id1,id2,...>` - run only selected experiments.

Example:

.. code-block:: bash

	vot evaluate --workspace ./workspace --force mytracker

Perform analysis
----------------

Compute analysis results from evaluated tracker outputs.

.. code-block:: bash

	vot analysis [--workspace <path>] [--format json|yaml] [--name <name>] [<tracker> ...]

Arguments and options:

* `[<tracker> ...]` - optional list of trackers (if omitted, available results are used),
* `--workspace <path>` - workspace directory,
* `--format json|yaml` - analysis output format (default: `json`),
* `--name <name>` - output name (default: generated timestamp).

Example:

.. code-block:: bash

	vot analysis --workspace ./workspace --format json mytracker

Generate a report
-----------------

Generate report files from analysis/evaluation data.

.. code-block:: bash

	vot report [--workspace <path>] [--format html|latex|plots] [--name <name>] [--sequences <ids>] [--experiments <ids>] [<tracker> ...]

Arguments and options:

* `[<tracker> ...]` - optional trackers to include,
* `--workspace <path>` - workspace directory,
* `--format html|latex|plots` - output type (default: `html`),
* `--name <name>` - output name,
* `--sequences <id1,id2,...>` - include only selected sequences,
* `--experiments <id1,id2,...>` - include only selected experiments.

Example:

.. code-block:: bash

	vot report --workspace ./workspace --format html mytracker

Pack tracker results
--------------------

Create a submission archive for a tracker. This command is needed for
participation in VOT challenges, but can also be used to create a standardized output archive for any tracker.

.. code-block:: bash

	vot pack [--workspace <path>] <tracker>

Arguments and options:

* `<tracker>` - tracker identifier,
* `--workspace <path>` - workspace directory.

Example:

.. code-block:: bash

	vot pack --workspace ./workspace mytracker