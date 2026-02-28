"""Multi-run experiments. This module contains the implementation of multi-run experiments. 
 Multi-run experiments are used to run a tracker multiple times on the same sequence. """
from typing import Callable

from attributee import Boolean, Integer, Float, List, String

from vot.experiment import Experiment
from vot.tracker import Tracker, Trajectory, ObjectQuery
from vot.tracker.online import OnlineTrackerRuntime
from vot.dataset import Sequence
from vot.region import Special, calculate_overlap
from vot.dataset.proxy import FrameMapSequence

class MultiRunExperiment(Experiment):
    """Base class for multi-run experiments. 
    Multi-run experiments are used to run a tracker multiple times on the same sequence."""

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
            prefix = sequence.name if not multiobject else f"{sequence.name}_{o}"
            for i in range(1, self.repetitions+1):
                name = f"{prefix}_{i:03d}"
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
            prefix = sequence.name if not multiobject else f"{sequence.name}_{o}"
            for i in range(1, self.repetitions+1):
                name =  f"{prefix}_{i:03d}"
                if Trajectory.exists(results, name):
                    trajectories.append(Trajectory.read(results, name))
                elif pad:
                    trajectories.append(None)
        return trajectories
    
    def execute(self, tracker, sequence, force = False, callback = None):
        raise NotImplementedError("This method should be implemented by subclasses.")

class UnsupervisedExperiment(MultiRunExperiment):
    """Unsupervised experiment. 
    This experiment is used to run a tracker multiple times on the same sequence without any supervision."""

    multiobject = Boolean(default=False)

    @property
    def _multiobject(self) -> bool:
        """Whether the experiment is multi-object or not.

        Returns:
            bool: True if the experiment is multi-object, False otherwise.
        """
        return self.multiobject

    def scan(self, tracker: Tracker, sequence: Sequence) -> tuple:
        """Scan the results of the experiment for the given tracker and sequence.
        
        Args:
            tracker (Tracker): The tracker to be scanned.
            sequence (Sequence): The sequence to be scanned.
            
        Returns:
            [tuple]: A tuple containing three elements. The first element is a boolean indicating whether the experiment is complete. The second element is a list of files that are present. The third element is the results object.
        """
        
        complete, files, results = super().scan(tracker, sequence)
        
        if results.exists(f"{sequence.name}_time.txt"):
            files.append(f"{sequence.name}_time.txt")
        elif complete:
            complete = False
        
        return complete, files, results

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
            return f"{sequence.name}_{o}_{i:03d}" if multiobject else f"{sequence.name}_{i:03d}"

        # Generate object queries for all objects in the sequence
        queries = []
        queries_keys = []
        for i in range(len(sequence)):
            for o in helper.new(i):
                queries.append(ObjectQuery(self._get_initialization(sequence, i, o), {}, i))
                queries_keys.append(o)

        with self._get_runtime(tracker, sequence, self._multiobject) as runtime:

            for i in range(1, self.repetitions+1):

                trajectories = {}

                times = []

                for o in helper.all(): 
                    trajectories[o] = Trajectory(len(sequence))

                if all([Trajectory.exists(results, result_name(sequence, o, i)) for o in trajectories.keys()]) and not force:
                    continue

                if self._can_stop(tracker, sequence):
                    return

                if runtime.multiobject:
                    
                    status = runtime.run(sequence, queries)
                    
                    for o, key in enumerate(queries_keys):
                        trajectories[key].set(0, Special(Trajectory.INITIALIZATION), status.objects[o][0].properties)
                    for frame in range(1, len(sequence)):
                        for o, key in enumerate(queries_keys):
                            trajectories[key].set(frame, status.objects[o][frame].region, status.objects[o][frame].properties)

                    times = status.times

                    if callback:
                        callback(float(i) / self.repetitions)
                        
                else:
                    
                    times = [0] * len(sequence)
                    
                    for q, query in enumerate(queries):
                        offset = query.offset
                        
                        proxy = FrameMapSequence(sequence, list(range(offset, len(sequence))))
                        status = runtime.run(proxy, [ObjectQuery(query.state, query.properties, 0)])
                        
                        trajectory = trajectories[queries_keys[q]]
                        trajectory.set(offset, Special(Trajectory.INITIALIZATION), status.objects[0][0].properties)
                        for frame in range(1, len(proxy)):
                            trajectory.set(frame + offset, status.objects[0][frame].region, status.objects[0][frame].properties)
                            times[frame + offset] += status.times[frame]
                            
                        if callback:
                            callback((float(i-1) / self.repetitions) + \
                                    (float(q) / (self.repetitions * len(trajectories))))
                    
                with results.write(f"{sequence.name}_time.txt") as filehandle:
                    filehandle.writelines([f"{t}\n" for t in times])
                        
                for o, trajectory in trajectories.items():
                    trajectory.write(results, result_name(sequence, o, i))


class SupervisedExperiment(MultiRunExperiment):
    """Supervised experiment. 
    This experiment is used to run a tracker multiple times 
    on the same sequence with supervision (reinitialization in case of failure).
    
    Due to the nature of the experiment, it requires online tracker runtimes and only
    works on single-target sequences. In all other cases the execution will fail with an error.
    """

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

        if len(sequence.objects()) != 1:
            raise ValueError("SupervisedExperiment only works on single-target sequences.")

        results = self.results(tracker, sequence)

        with self._get_runtime(tracker, sequence) as runtime:

            if not isinstance(runtime, OnlineTrackerRuntime):
                raise ValueError("SupervisedExperiment requires an online tracker runtime.")

            for i in range(1, self.repetitions+1):
                name = f"{sequence.name}_{i:03d}"

                if Trajectory.exists(results, name) and not force:
                    continue

                if self._can_stop(tracker, sequence):
                    return

                trajectory = Trajectory(len(sequence))

                frame = 0
                while frame < len(sequence):

                    _, elapsed = runtime.initialize(sequence.frame(frame), self._get_initialization(sequence, frame))

                    trajectory.set(frame, Special(Trajectory.INITIALIZATION), {"time" : elapsed})

                    frame = frame + 1

                    while frame < len(sequence):

                        target, elapsed = runtime.update(sequence.frame(frame))

                        target.properties["time"] = elapsed

                        if calculate_overlap(target.region, sequence.groundtruth(frame), sequence.size) <= self.failure_overlap:
                            trajectory.set(frame, Special(SupervisedExperiment.FAILURE), target.properties)
                            frame = frame + self.skip_initialize
 
                            if self.skip_tags:
                                while frame < len(sequence):
                                    if not [t for t in sequence.tags(frame) if t in self.skip_tags]:
                                        break
                                    frame = frame + 1
                            break
                        else:
                            trajectory.set(frame, target.region, target.properties)
                        frame = frame + 1

                if  callback:
                    callback(i / self.repetitions)

                trajectory.write(results, name)
