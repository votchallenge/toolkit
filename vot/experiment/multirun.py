"""Multi-run experiments. This module contains the implementation of multi-run experiments. 
 Multi-run experiments are used to run a tracker multiple times on the same sequence. """
from typing import Callable

from vot.dataset import Sequence
from vot.region import Special, calculate_overlap

from attributee import Boolean, Integer, Float, List, String

from vot.experiment import Experiment, experiment_registry
from vot.tracker import Tracker, Trajectory, ObjectStatus

class MultiRunExperiment(Experiment):
    """Base class for multi-run experiments. Multi-run experiments are used to run a tracker multiple times on the same sequence."""

    repetitions = Integer(val_min=1, default=1)
    early_stop = Boolean(default=True)

    def _can_stop(self, tracker: Tracker, sequence: Sequence):
        """Check whether the experiment can be stopped early.
        
        Args:
            tracker (Tracker): The tracker to be checked.
            sequence (Sequence): The sequence to be checked.
            
        Returns:
            bool: True if the experiment can be stopped early, False otherwise.
        """
        if not self.early_stop:
            return False
        
        for o in sequence.objects():
            trajectories = self.gather(tracker, sequence, objects=[o])
            if len(trajectories) < 3:
                return False
            for trajectory in trajectories[1:]:
                if not trajectory.equals(trajectories[0]):
                    return False

        return True

    def scan(self, tracker: Tracker, sequence: Sequence):
        """Scan the results of the experiment for the given tracker and sequence.

        Args:
            tracker (Tracker): The tracker to be scanned.
            sequence (Sequence): The sequence to be scanned.

        Returns:
            [tuple]: A tuple containing three elements. The first element is a boolean indicating whether the experiment is complete. The second element is a list of files that are present. The third element is the results object.
        """
        
        results = self.results(tracker, sequence)

        files = []
        complete = True
        multiobject = len(sequence.objects()) > 1
        assert self._multiobject or not multiobject

        for o in sequence.objects():
            prefix = sequence.name if not multiobject else "%s_%s" % (sequence.name, o)
            for i in range(1, self.repetitions+1):
                name = "%s_%03d" % (prefix, i)
                if Trajectory.exists(results, name):
                    files.extend(Trajectory.gather(results, name))
                elif self._can_stop(tracker, sequence):
                    break
                else:
                    complete = False
                    break

        return complete, files, results

    def gather(self, tracker: Tracker, sequence: Sequence, objects = None, pad = False):
        """Gather trajectories for the given tracker and sequence.
        
        Args:
            tracker (Tracker): The tracker to be used.
            sequence (Sequence): The sequence to be used.
            objects (list, optional): The list of objects to be gathered. Defaults to None.
            pad (bool, optional): Whether to pad the list of trajectories with None values. Defaults to False.
            
        Returns:
            list: The list of trajectories.
        """
        trajectories = list()

        multiobject = len(sequence.objects()) > 1

        assert self._multiobject or not multiobject
        results = self.results(tracker, sequence)

        if objects is None:
            objects = list(sequence.objects())

        for o in objects:
            prefix = sequence.name if not multiobject else "%s_%s" % (sequence.name, o)
            for i in range(1, self.repetitions+1):
                name =  "%s_%03d" % (prefix, i)
                if Trajectory.exists(results, name):
                    trajectories.append(Trajectory.read(results, name))
                elif pad:
                    trajectories.append(None)
        return trajectories

@experiment_registry.register("unsupervised")
class UnsupervisedExperiment(MultiRunExperiment):
    """Unsupervised experiment. This experiment is used to run a tracker multiple times on the same sequence without any supervision."""

    multiobject = Boolean(default=False)

    @property
    def _multiobject(self) -> bool:
        """Whether the experiment is multi-object or not.

        Returns:
            bool: True if the experiment is multi-object, False otherwise.
        """
        return self.multiobject

    def execute(self, tracker: Tracker, sequence: Sequence, force: bool = False, callback: Callable = None):
        """Execute the experiment for the given tracker and sequence.

        Args:
            tracker (Tracker): The tracker to be used.
            sequence (Sequence): The sequence to be used.
            force (bool, optional): Whether to force the execution. Defaults to False.
            callback (Callable, optional): The callback to be used. Defaults to None.
        """

        from .helpers import MultiObjectHelper

        results = self.results(tracker, sequence)

        multiobject = len(sequence.objects()) > 1
        assert self._multiobject or not multiobject

        helper = MultiObjectHelper(sequence)

        def result_name(sequence, o, i):
            """Get the name of the result file."""
            return "%s_%s_%03d" % (sequence.name, o, i) if multiobject else "%s_%03d" % (sequence.name, i)

        with self._get_runtime(tracker, sequence, self._multiobject) as runtime:

            for i in range(1, self.repetitions+1):

                trajectories = {}

                for o in helper.all(): trajectories[o] = Trajectory(sequence.length)

                if all([Trajectory.exists(results, result_name(sequence, o, i)) for o in trajectories.keys()]) and not force:
                    continue

                if self._can_stop(tracker, sequence):
                    return

                _, elapsed = runtime.initialize(sequence.frame(0), [ObjectStatus(self._get_initialization(sequence, 0, x), {}) for x in helper.new(0)])

                for x in helper.new(0):
                    trajectories[x].set(0, Special(Trajectory.INITIALIZATION), {"time": elapsed})

                for frame in range(1, sequence.length):
                    state, elapsed = runtime.update(sequence.frame(frame), [ObjectStatus(self._get_initialization(sequence, 0, x), {}) for x in helper.new(frame)])

                    if not isinstance(state, list):
                        state = [state]

                    for x, object in zip(helper.objects(frame), state):
                        object.properties["time"] = elapsed # TODO: what to do with time stats?
                        trajectories[x].set(frame, object.region, object.properties)

                    if callback:
                        callback(float(i-1) / self.repetitions + (float(frame) / (self.repetitions * len(sequence))))

                for o, trajectory in trajectories.items():
                    trajectory.write(results, result_name(sequence, o, i))


@experiment_registry.register("supervised")
class SupervisedExperiment(MultiRunExperiment):
    """Supervised experiment. This experiment is used to run a tracker multiple times on the same sequence with supervision (reinitialization in case of failure)."""

    FAILURE = 2

    skip_initialize = Integer(val_min=1, default=1)
    skip_tags = List(String(), default=[])
    failure_overlap = Float(val_min=0, val_max=1, default=0)

    def execute(self, tracker: Tracker, sequence: Sequence, force: bool = False, callback: Callable = None):
        """Execute the experiment for the given tracker and sequence.

        Args:
            tracker (Tracker): The tracker to be used.
            sequence (Sequence): The sequence to be used.
            force (bool, optional): Whether to force the execution. Defaults to False.
            callback (Callable, optional): The callback to be used. Defaults to None.
        """

        results = self.results(tracker, sequence)

        with self._get_runtime(tracker, sequence) as runtime:

            for i in range(1, self.repetitions+1):
                name = "%s_%03d" % (sequence.name, i)

                if Trajectory.exists(results, name) and not force:
                    continue

                if self._can_stop(tracker, sequence):
                    return

                trajectory = Trajectory(sequence.length)

                frame = 0
                while frame < sequence.length:

                    _, elapsed = runtime.initialize(sequence.frame(frame), self._get_initialization(sequence, frame))

                    trajectory.set(frame, Special(Trajectory.INITIALIZATION), {"time" : elapsed})

                    frame = frame + 1

                    while frame < sequence.length:

                        object, elapsed = runtime.update(sequence.frame(frame))

                        object.properties["time"] = elapsed

                        if calculate_overlap(object.region, sequence.groundtruth(frame), sequence.size) <= self.failure_overlap:
                            trajectory.set(frame, Special(SupervisedExperiment.FAILURE), object.properties)
                            frame = frame + self.skip_initialize
 
                            if self.skip_tags:
                                while frame < sequence.length:
                                    if not [t for t in sequence.tags(frame) if t in self.skip_tags]:
                                        break
                                    frame = frame + 1
                            break
                        else:
                            trajectory.set(frame, object.region, object.properties)
                        frame = frame + 1

                if  callback:
                    callback(i / self.repetitions)

                trajectory.write(results, name)
