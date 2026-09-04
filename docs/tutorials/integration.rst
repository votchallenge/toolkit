Tracker integration
===================

This tutorial shows how to integrate a tracker into the toolkit in two steps:

* prepare an executable tracker wrapper,
* register it in a tracker registry file (`trackers.ini`).

Goal
----

At the end, you will be able to run:

.. code-block:: bash

	vot test <tracker-id>

and the toolkit will execute your tracker through the selected protocol.

Step 1: Choose protocol
-----------------------

The toolkit supports multiple runtime protocols. The most commonly used are:

* `trax` (and adapters `traxpython`, `traxmatlab`, `traxoctave`) for online tracking communication,
* `folder` (and adapters `folderpython`, `foldermatlab`, `folderoctave`) for file-based batch execution,
* `python` for direct Python-process execution.

Pick the protocol that matches your wrapper implementation.

Step 2: Prepare tracker wrapper
-------------------------------

Create a wrapper script or executable that can be launched by a shell command. The command must be stable (same executable, same dependencies) and should exit with non-zero status on failure.

For example, if your wrapper is a Python script:

.. code-block:: text

	wrappers/my_tracker.py

you may run it with command:

.. code-block:: bash

	python wrappers/my_tracker.py

If your wrapper needs project-local modules, make sure they are importable from the execution environment.

Step 3: Create a registry file
------------------------------

Create either `trackers.ini` or `trackers.yaml` in the workspace root (the same
directory where you run `vot` commands). When a registry path points to a
directory, the toolkit looks for both files automatically.

.. code-block:: text

	workspace/trackers.ini

Add one section per tracker. Section name is the tracker identifier used in CLI commands.

Minimal example:

.. code-block:: ini

	[mytracker]
	label = My Tracker
	protocol = trax
	command = python wrappers/my_tracker.py

Field reference:

* `label` - display name in reports,
* `protocol` - runtime protocol identifier,
* `command` - shell command used to launch the tracker.

Optional fields:

* `tags = fast,baseline` - comma-separated tags,
* `env_PATH = /opt/mydeps/bin:${PATH}` - environment variables (`env_<NAME>`),
* `arg_model = /models/model.onnx` - protocol arguments (`arg_<name>`),
* protocol-specific options, e.g. `timeout = 60` or `convert = mask` (for folder-based conversion).

YAML registry
~~~~~~~~~~~~~

The YAML format uses the tracker identifier as the top-level key. The fields are
the same as for an INI entry:

.. code-block:: yaml

	mytracker:
	  label: My Tracker
	  protocol: trax
	  command: python wrappers/my_tracker.py
	  timeout: 60
	  tags:
	    - fast
	    - baseline
	  env:
	    PYTHONPATH: /opt/mytracker
	  arguments:
	    model: /models/model.onnx

For simple entries, `env_<NAME>` and `arg_<name>` fields can also be used in
YAML, just as in INI files. Tracker identifiers must contain only letters,
numbers, hyphens, and underscores.

Step 4: Verify registry loading
-------------------------------

From the workspace directory, run:

.. code-block:: bash

	vot test mytracker

If registry loading and wrapper execution are correct, the toolkit runs an integration test on a synthetic sequence.

Useful checks:

* `vot test -g mytracker` to visualize test output,
* `vot --registry /path/to/another/registry test mytracker` to use a custom registry location,
* `vot --debug test mytracker` for detailed logs.

Step 5: Run in workspace pipeline
---------------------------------

After successful testing, run regular workspace commands:

.. code-block:: bash

	vot evaluate --workspace <workspace-path> mytracker

Then (if your stack supports local analysis):

.. code-block:: bash

	vot analysis --workspace <workspace-path> mytracker
	vot report --workspace <workspace-path> mytracker

Troubleshooting
---------------

* **Tracker not found**: confirm section name in `trackers.ini` matches the CLI tracker id.
* **Protocol not available**: verify `protocol` value is one of registered runtime protocols.
* **Process fails to start**: run the `command` manually in terminal to validate interpreter, paths, and dependencies.
* **Imports fail**: configure `env_PYTHONPATH` or use absolute paths in `command`.
* **Timeouts**: increase `timeout` in tracker registry entry.