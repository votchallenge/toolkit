Experiment stack
================

The experiment stack is a collection of experiments that are run on the dataset. 
It is described by a YAML file that may be provided in the toolkit 
installation or created by the user manually in the workspace.

Stack file structure
--------------------

* ``title`` (string, optional)

	Human-readable stack title. Default is ``"Stack"``. Not really needed, but used to make the stack more identifiable and to provide a default title for reports.

* ``dataset`` (string, optional)

	Dataset locator used by the stack. Usually a URL to a dataset description or
	archive. If omitted, default is ``null``. If URL is present, the toolkit will
    try to download sequences from the locaton when creating a new workspace.
    If you are creating a workspace manually with sequences already present, the field 
    is not needed.

* ``url`` (string, optional)

	Reference URL for the challenge/stack page. Default is an empty string. Used for challenges, but otherwise only informative.

* ``experiments`` (mapping, required)

	Mapping from experiment identifier to and experiment to be performed.
	The mapping key is used as the internal experiment identifier.

    Each experiment requires a ``type``, which specifies the type of the experiment,
    based on its type, more parameters are possible.

    Each experiment can also specify a set of core performance analyses associated with the
    experiment in the ``analyses`` field. See sections below for more details.

Minimal template
----------------

Use the following YAML template as a starting point:

.. code-block:: yaml

	experiments:
		base:
			type: unsupervised
			repetitions: 1

Extended template with analyses for experiments
-----------------------------------------------

If you need more than a minimal stack, start from this pattern and fill values
manually where indicated:

.. code-block:: yaml

	title: "Stack Example"
	dataset: https://example.net/data/dataset.zip
	experiments:
		base:
			type: unsupervised
			repetitions: 5
			analyses:
			    - type: <manual-entry>
				  name: <manual-entry>
				  ... more arguments


