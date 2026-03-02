Creating a custom dataset
=========================

This tutorial shows how to create a minimal dataset in the default toolkit format and use it inside a workspace.
Note that this is only needed if you want to use your own dataset. If you just want to use one of the VOT challenges or benchmarks, the
dataset will be automatically downloaded and prepared for you when you initialize the workspace with the appropriate stack. 

Goal
----

At the end, you will have:

* one sequence in the correct format,
* a dataset index file,
* an initialized workspace that uses your local data.

Step 1: Prepare directories
---------------------------

Create a workspace root and a sequence directory:

.. code-block:: bash

	mkdir -p myworkspace/sequences/mysequence/color

The sequence name is the directory name (`mysequence`).

Step 2: Add frames
------------------

Put all images into the channel folder and name them with 8-digit indices:

.. code-block:: text

	myworkspace/sequences/mysequence/color/00000001.jpg
	myworkspace/sequences/mysequence/color/00000002.jpg
	myworkspace/sequences/mysequence/color/00000003.jpg
	...

The default loader expects this numeric ordering. Supported channels are typically `color`, `depth`, and `ir`.

Step 3: Add annotations
-----------------------

For a single-object sequence, create `groundtruth.txt` in the sequence root:

.. code-block:: text

	myworkspace/sequences/mysequence/groundtruth.txt

Write one region per frame (for example rectangle format `x,y,w,h`), one line per frame:

.. code-block:: text

	120,80,64,92
	121,81,64,92
	122,81,65,93

For multi-object sequences, use one file per object:

.. code-block:: text

	groundtruth_target1.txt
	groundtruth_target2.txt

Step 4: Create sequence metadata
--------------------------------

Create a file named `sequence` inside the sequence directory:

.. code-block:: text

	myworkspace/sequences/mysequence/sequence

Use key-value entries:

.. code-block:: ini

	channel.default=color
	channels.color=color/%08d.jpg
	fps=30

Optional fields such as `width`, `height`, and `length` can be added, but are not required for a basic setup.

Step 5: (Optional) Add tags and values
--------------------------------------

You can attach per-frame metadata:

* `<name>.tag` (binary flags, one line per frame, values `0` or `1`)
* `<name>.value` (floating-point values, one line per frame)

Example:

.. code-block:: text

	myworkspace/sequences/mysequence/occlusion.tag
	myworkspace/sequences/mysequence/confidence.value

Step 6: Index the dataset
-------------------------

Create `list.txt` in `sequences` and list one sequence name per line:

.. code-block:: text

	myworkspace/sequences/list.txt

.. code-block:: text

	mysequence

Step 7: Initialize workspace using local dataset
------------------------------------------------

Initialize the workspace with a stack and disable dataset download:

.. code-block:: bash

	vot initialize <stack-name> --workspace myworkspace --nodownload

This creates workspace configuration while keeping your local dataset in `sequences/`.

Step 8: Verify
--------------

Check that files exist:

.. code-block:: text

	myworkspace/
	├── config.yaml
	├── sequences/
	│   ├── list.txt
	│   └── mysequence/
	│       ├── sequence
	│       ├── groundtruth.txt
	│       └── color/
	│           ├── 00000001.jpg
	│           └── ...
	├── results/
	├── analysis/
	└── cache/

See `dataset specification <../dataset.md#default-dataset-format>`_ for more details on the expected format.