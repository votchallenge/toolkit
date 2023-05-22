Overview
========

The toolkit is designed as a modular framework with several modules that address different aspects of the performance evaluation problem.

Key concepts
------------

Key concepts that are used throughout the toolkit are:

* **Dataset** - a collection of sequences that is used for performance evaluation. A dataset is a collection of **sequences**.
* **Sequence** - a sequence of frames with correspoding ground truth annotations for one or more objects. A sequence is a collection of **frames**.
* **Tracker** - a tracker is an algorithm that takes frames from a sequence as input (one by one) and produces a set of **trajectories** as output.
* **Experiment** - an experiment is a method that applies a tracker to a given sequence in a specific way.
* **Analysis** - an analysis is a set of **measures** that are used to evaluate the performance of a tracker (compare predicted trajectories to groundtruth).
* **Stack** - a stack is a collection of **experiments** and **analyses** that are performed on a given dataset.
* **Workspace** - a workspace is a collection of experiments and analyses that are performed on a given dataset.

Tracker support
---------------

The toolkit supports various ways of interacting with a tracking methods. Primary manner (at the only supported at the moment) is using the TraX protocol. 
The toolkit provides a wrapper for the TraX protocol that allows to use any tracker that supports the protocol.

Dataset support
---------------

The toolkit is capable of using any dataset that is provided in the toolkit format. 
The toolkit format is a simple directory structure that contains a set of sequences. Each sequence is a directory that contains a set of frames and a groundtruth file. 
The groundtruth file is a text file that contains one line per frame. Each line contains the bounding box of the object in the frame in the format `x,y,w,h`. The toolkit format is used by the toolkit itself and by the VOT challenges.


Performance methodology support
-------------------------------

Various performance measures and visualzatons are implemented, most of them were used in VOT challenges.

 * **Accuracy** - the accuracy measure is the overlap between the predicted and groundtruth bounding boxes. The overlap is measured using the intersection over union (IoU) measure.
 * **Robustness** - the robustness measure is the number of failures of the tracker. A failure is defined as the overlap between the predicted and groundtruth bounding boxes being less than 0.5.
 * **Expected Average Overlap** - the expected average overlap (EAO) is a measure that combines accuracy and robustness into a single measure. The EAO is computed as the area under the accuracy-robustness curve.
 * **Expected Overlap** - the expected overlap (EO) is a measure that combines accuracy and robustness into a single measure. The EO is computed as the area under the accuracy-robustness curve.
