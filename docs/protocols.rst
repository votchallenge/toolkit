Trackers Protocols
==================

This section describes the protocols that trackers must follow to be compatible with the toolkit.

TraX protocol
-------------

In the toolkit, TraX is the primary online protocol for tracker integration.
It is standardized communication protocol for visual object tracking, more information
is available at `TraX Reference Implementation <https://trax.readthedocs.com>`_.

TraX is the preferred protocol for online trackers because it provides a
well-defined communication contract and robust runtime separation.
The toolkit starts the tracker process, exchanges frame/object messages with it,
and collects tracker outputs frame-by-frame.

Key properties:

* online communication (initialize/update style execution),
* process isolation between toolkit and tracker,
* support for tracker metadata and runtime options,
* automatic handling of supported region/image channels through TraX bindings.

Common runtime variants:

* ``trax`` - native executable speaking TraX,
* ``traxpython`` - Python adapter wrapper,
* ``traxmatlab`` - Matlab adapter wrapper,
* ``traxoctave`` - Octave adapter wrapper.

Typical tracker registry entry:

.. code-block:: ini

    [mytraxtracker]
    label = My TraX Tracker
    protocol = trax
    command = /path/to/tracker_executable
    timeout = 30

Folder protocol
---------------

The folder protocol is a simple file-based protocol for trackers that read and write data
from a specified folder. It is designed to be easy to implement and use, and is suitable
for trackers that do not require real-time performance. A tracker is essentially run as
a batch program over a directory of structured data.
The tracker then outputs the required output in the specified files.

Sequence specification
~~~~~~~~~~~~~~~~~~~~~~

The sequence is specified as a sequence of image files.
Multiple input channels are supported, depending on the sequence.
The file is ``frames_<CHANNEL>.txt``, and each line contains a single path to a frame file.
This can be an absolute or a relative path (in this case, relative to the working directory).

Query specification
~~~~~~~~~~~~~~~~~~~

A query is specified in a single file per object.
All files follow the naming pattern ``query_<ID>.txt``, where ``<ID>`` denotes the string identifier of an object (alphanumeric sequence). The file contains the following lines:

* Offset - a single value for the temporal location of the query; frames start with number 0.
* State - Initialization state, a comma-separated sequence of numbers. Can contain various state formats:
    * No state - single ``0`` (can be used for referral and specify description via a text argument)
    * Point - two numbers
    * Rectangle - four numbers
    * Polygon - six or more (even) numbers
    * Mask - using the same format as the toolkit is using (code from the toolkit can be used)
* Additional lines contain optional arguments in the form of ``key=value``

Output trajectories
~~~~~~~~~~~~~~~~~~~

At the end of the tracking process, the tracker should output a sequence of files.
For each query object, the mandatory file is ``output_<ID>.txt``, which contains the object states, one line per frame in the sequence.
This means that if the object is not present in a given frame or was even queried at a later frame, the tracker should output ``0`` for that frame.

Additionally, the tracker may return optional values for the object in the form of ``output_<ID>_<VALUE>.txt``, where each frame's values are provided.

Tracker metadata specifics
~~~~~~~~~~~~~~~~~~~~~~~

Since the folder protocols does not support automatic region type reporting and conversion as in TraX, this 
has to be specified in the tracker metadata (registry). You can specify automatic conversion with
the ``convert`` field, can be one of the following: ``rectangle``, ``polygon``, ``mask`` or ``point``.
This way, the tracker will get an appropriate region specification via automatic conversion.

Tracker metadata example
~~~~~~~~~~~~~~~~~~~~~~~~

Example ``trackers.ini`` entry for the folder protocol:

.. code-block:: ini

    [myfoldertracker]
    label = My Folder Tracker
    protocol = folder
    command = python /path/to/run_folder_tracker.py
    timeout = 120
    convert = rectangle

    ; optional environment variables
    env_PYTHONPATH = /path/to/project:${PYTHONPATH}

    ; optional tracker arguments forwarded by the runtime
    arg_model = /path/to/model.onnx
    arg_device = cuda

Python protocol
---------------

The Python protocol is a lightweight runtime for Python-native trackers. The toolkit runs tracker code in a separate process using the standard ``multiprocessing`` module and communicates with it using input/output task queues.

Compared to TraX, this protocol is intentionally minimal and Python-only:

* no transport layer setup,
* no region format conversion,
* direct method calls on a Python tracker object in an isolated process.

Tracker object interface
~~~~~~~~~~~~~~~~~~~~~~~~

The instantiated tracker object should expose:

* initialization method: ``init(...)`` or ``initialize(...)``,
* update method: ``update(...)``.

Method names can be overridden in tracker metadata using ``initialize_method`` and ``update_method``.

Command resolution
~~~~~~~~~~~~~~~~~~

For the Python protocol, the ``command`` field is interpreted as an import path to a factory/class:

* ``mypackage.mytracker:Tracker``
* ``mypackage.mytracker.Tracker``
* ``mypackage.mytracker`` (module fallback to ``create_tracker`` or ``Tracker``)

If constructor arguments are needed, specify them as tracker metadata ``arg_<name> = <value>``. They are forwarded as keyword arguments when constructing the tracker object.

Example registry entry
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

    [mypythontracker]
    label = My Python Tracker
    protocol = python
    command = mypackage.mytracker:Tracker
    timeout = 60
    initialize_method = init
    update_method = update
    arg_model = /models/model.onnx

Execution model
~~~~~~~~~~~~~~~

The runtime process handles the following queue tasks:

* ``initialize`` - initialize tracker on first frame/query state,
* ``update`` - update tracker on subsequent frames,
* ``stop`` - terminate worker process.

Each task returns tracker outputs and elapsed time through the output queue.

Notes
~~~~~

* Process isolation improves robustness and security compared to in-process execution.
* The protocol is intended for Python trackers only.
* Tracker methods should return structures compatible with toolkit runtime expectations.