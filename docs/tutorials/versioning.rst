Tracker versioning
==================

This tutorial explains how tracker references are versioned in the toolkit and how to avoid conflicts when you run multiple configurations or multiple codebases.

Goal
----

At the end, you will be able to:

* run one tracker identifier with multiple versions,
* map different initialization parameters to separate tracker versions,
* avoid collisions when different codebases use the same tracker identifier.

API behavior summary
--------------------

Versioning behavior is defined by `vot.tracker.Registry` and `vot.tracker.Tracker`:

* tracker references use the form `identifier@version`,
* `Registry.resolve("identifier@version")` returns that specific version,
* `Registry.resolve("identifier@", storage=...)` resolves all available versions found in results storage,
* `Tracker.reversion(version)` creates a versioned view of the same tracker definition,
* tracker identifiers must be unique inside a resolved registry set.

Important: if the same identifier is found more than once while scanning registry files, only the first one is kept and later duplicates are ignored.

Reference syntax
----------------

Use the following forms in CLI commands that accept tracker references (`evaluate`, `analysis`, `report`):

* `mytracker` - base tracker reference,
* `mytracker@ablationA` - explicit version,
* `mytracker@` - all discovered versions for `mytracker` in workspace results.

The wildcard form (`@` with empty version) depends on existing result folders and therefore is most useful after at least one evaluation run.

Case 1: One tracker, multiple parameter sets
--------------------------------------------

A common pattern is one executable with different initialization parameters (for example model path, search size, or update interval).

Create separate registry entries with distinct identifiers:

.. code-block:: ini

	[ostrack_small]
	label = OSTrack (small)
	protocol = trax
	command = python run.py
	arg_model = models/ostrack_small.onnx
	arg_search = 256

	[ostrack_large]
	label = OSTrack (large)
	protocol = trax
	command = python run.py
	arg_model = models/ostrack_large.onnx
	arg_search = 384

Evaluate both:

.. code-block:: bash

	vot evaluate --workspace ./workspace ostrack_small ostrack_large

This is the safest approach when parameter sets should be treated as separate trackers in reports.

Case 2: One identifier, explicit version labels
------------------------------------------------

If you need to keep a single base identifier but still separate runs, use versioned references in commands.

Example:

.. code-block:: bash

	vot evaluate --workspace ./workspace mytracker@default
	vot evaluate --workspace ./workspace mytracker@long_window
	vot evaluate --workspace ./workspace mytracker@no_redetect

Then analyze all discovered versions at once:

.. code-block:: bash

	vot analysis --workspace ./workspace mytracker@
	vot report --workspace ./workspace mytracker@

Practical recommendation: ensure your wrapper behavior is deterministic for each version label (for example via version-aware command-line options or environment variables) so results remain reproducible.

Case 3: Same identifier in different codebases
----------------------------------------------

When two codebases expose the same identifier (for example both define `trackerx`), identifier collision happens during registry loading.

Because registry loading keeps the first identifier and ignores later duplicates, avoid ambiguous setups.

Preferred options:

* rename identifiers to include codebase namespace,
* keep one identifier and encode codebase in version suffix,
* isolate each codebase in a separate workspace if strict separation is required.

Namespaced identifier example:

.. code-block:: ini

	[trackerx_implA]
	label = TrackerX (implementation A)
	protocol = trax
	command = python /path/codebaseA/run.py

	[trackerx_implB]
	label = TrackerX (implementation B)
	protocol = trax
	command = python /path/codebaseB/run.py

Single identifier with versions example:

.. code-block:: bash

	vot evaluate --workspace ./workspace trackerx@implA
	vot evaluate --workspace ./workspace trackerx@implB

Registry ordering notes
-----------------------

Registry paths come from workspace `config.yaml` (`registry` list) and optional CLI `--registry` arguments.

During scan:

* directory paths are expanded to `trackers.yaml` and `trackers.ini`,
* files are processed in order,
* duplicate identifiers are skipped after first occurrence.

Therefore, ordering of registry locations matters whenever identifiers overlap.

Python API examples
-------------------

Resolve explicit versions:

.. code-block:: python

	from vot.tracker import Registry
	from vot.workspace import Workspace

	workspace = Workspace.load("./workspace")
	registry = workspace.registry
	results = workspace.storage.substorage("results")

	trackers = registry.resolve(
	    "mytracker@default",
	    "mytracker@long_window",
	    storage=results,
	    skip_unknown=False,
	)

Resolve all discovered versions:

.. code-block:: python

	all_versions = registry.resolve("mytracker@", storage=results, skip_unknown=False)
	for tracker in all_versions:
	    print(tracker.reference, tracker.label)

Troubleshooting
---------------

* **Version not resolved**: check that version name uses valid identifier characters (`a-z`, `A-Z`, `0-9`, `-`, `_`).
* **`mytracker@` returns empty list**: no matching result folders exist yet; run evaluation for at least one version first.
* **Unexpected tracker implementation used**: look for duplicate identifiers across registry files and adjust registry ordering or rename identifiers.
* **Inconsistent results between runs**: verify that version labels map to fixed tracker parameters and fixed dependencies.