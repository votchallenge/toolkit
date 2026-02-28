from typing import Tuple, List, Dict
from abc import abstractmethod

from vot.region import Special
from vot.tracker.results import Trajectory
from vot.tracker import Tracker, TrackerRuntime, FrameObjects, Queries, ObjectStatus, RunStatus
from vot.dataset import Frame

class OnlineTrackerRuntime(TrackerRuntime):
    """Base class for online tracker runtime implementations. 
    Tracker runtime is responsible for running the tracker executable 
    and communicating with it."""

    def __init__(self, tracker: Tracker):
        """Creates a new tracker runtime instance.

        Args:
            tracker (Tracker): The tracker instance.
        """
        self._tracker = tracker

    @property
    def tracker(self) -> Tracker:
        """Returns the tracker instance associated with this runtime."""
        return self._tracker

    def __enter__(self):
        """Starts the tracker runtime."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stops the tracker runtime."""
        self.stop()

    @property
    def multiobject(self):
        """Returns True if the tracker supports multiple objects, False otherwise."""
        return False

    @abstractmethod
    def stop(self):
        """Stops the tracker runtime."""
        raise NotImplementedError

    @abstractmethod
    def restart(self):
        """Restarts the tracker runtime, usually stars a new process."""
        raise NotImplementedError

    @abstractmethod
    def initialize(self, frame: Frame, new: FrameObjects = None, properties: dict = None) -> Tuple[FrameObjects, float]:
        """Initializes the tracker runtime with specified frame and objects. Returns the initial objects and the time it took to initialize the tracker.
        
        Arguments:
            frame (Frame) -- The frame to initialize the tracker with.
            new (Objects) -- The objects to initialize the tracker with.
            properties (dict) -- The properties to initialize the tracker with.

        Returns:
            Tuple[Objects, float] -- The initial objects and the time it took to initialize the tracker.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, frame: Frame, new: FrameObjects = None, properties: dict = None) -> Tuple[FrameObjects, float]:
        """Updates the tracker runtime with specified frame and objects. Returns the updated objects and the time it took to update the tracker.

        Arguments:
            frame (Frame) -- The frame to update the tracker with.
            new (Objects) -- The objects to update the tracker with.
            properties (dict) -- The properties to update the tracker with.

        Returns:
            Tuple[Objects, float] -- The updated objects and the time it took to update the tracker.
        """
        raise NotImplementedError

    def run(self, frames: List[Frame], queries: Queries) -> RunStatus:
        """Runs the tracker on the given frames and queries. 
        Returns the tracker output as a RunStatus namedtuple.
        The online tracker runtime uses the interface defined by 
        the initialize and update methods to run the tracker 
        on the given frames and queries.

        Arguments:
            frames (List[Frame]) -- The list of frames to run the tracker on.
            queries (List[ObjectQuery]) -- The list of object queries to run the tracker on.
        """
        
        # Order the queries by offset and id
        
        statuses = []
        # Initialize statuses with empty lists for each query
        for i in range(len(queries)):
            statuses.append([])
        
        times = []
        
        for i, frame in enumerate(frames):
            # Filter out objects appearing in the current frame
            new = [ObjectStatus(queries[j].state, queries[j].properties) for j in range(len(queries)) if queries[j].offset == i]
        
            if i == 0:
                status, time = self.initialize(frames[0], new)
            else:
                status, time = self.update(frame, new)
        
            times.append(time)
        
            for j in range(len(queries)):
                if queries[j].offset <= i:
                    statuses[j].append(status[j])
                else:
                    statuses[j].append(ObjectStatus(Special(Trajectory.UNKNOWN), {})) 
                   
        return RunStatus(statuses, times)

class RealtimeTrackerRuntime(TrackerRuntime):
    """Base class for realtime tracker runtime implementations. 
    Realtime tracker runtime is responsible for running the tracker executable and communicating with it while simulating given real-time constraints."""

    def __init__(self, runtime: TrackerRuntime, grace: int = 1, interval: float = 0.1):
        """Initializes the realtime tracker runtime with specified tracker runtime, grace period and update interval.
        
        Arguments: 
            runtime {OnlineTrackerRuntime} -- The tracker runtime to wrap.
            grace {int} -- The grace period in seconds. The tracker will be updated at least once during the grace period. (default: {1})
            interval {float} -- The update interval in seconds. (default: {0.1})
            
        """
        if not isinstance(runtime, OnlineTrackerRuntime):
            raise ValueError("Runtime does not support online communication")
        
        super().__init__(runtime.tracker)
        self._runtime = runtime
        self._grace = grace
        self._interval = interval
        self._countdown = 0
        self._time = 0
        self._status = None

    @property
    def multiobject(self):
        """Returns True if the tracker supports multiple objects, False otherwise."""
        return self._runtime.multiobject

    def stop(self):
        """Stops the tracker runtime."""
        self._runtime.stop()
        self._time = 0
        self._status = None

    def restart(self):
        """Restarts the tracker runtime, usually stars a new process."""
        self._runtime.restart()
        self._time = 0
        self._status = None

    def initialize(self, frame: Frame, new: FrameObjects = None, properties: dict = None) -> Tuple[FrameObjects, float]:
        """Initializes the tracker runtime with specified frame and objects. Returns the initial objects and the time it took to initialize the tracker.
        
        Arguments:
            frame {Frame} -- The frame to initialize the tracker with.
            new {Objects} -- The objects to initialize the tracker with.
            properties {dict} -- The properties to initialize the tracker with.
            
        Returns:
            Tuple[Objects, float] -- The initial objects and the time it took to initialize the tracker.
        """
        self._countdown = self._grace
        self._status = None

        status, time = self._runtime.initialize(frame, new, properties)

        if time > self._interval:
            if self._countdown > 0:
                self._countdown = self._countdown - 1
                self._time = 0
            else:
                self._time = time - self._interval
                self._status = status
        else:
            self._time = 0

        return status, time


    def update(self, frame: Frame, _: FrameObjects = None, properties: dict = None) -> Tuple[FrameObjects, float]:
        """Updates the tracker runtime with specified frame and objects. Returns the updated objects and the time it took to update the tracker.
        
        Arguments:
            frame {Frame} -- The frame to update the tracker with.
            new {Objects} -- The objects to update the tracker with.
            properties {dict} -- The properties to update the tracker with.
            
        Returns:
            Tuple[Objects, float] -- The updated objects and the time it took to update the tracker.
        """

        if self._time > self._interval:
            self._time = self._time - self._interval
            return self._status, 0
        else:
            self._status = None
            self._time = 0

        status, time = self._runtime.update(frame, properties)

        if time > self._interval:
            if self._countdown > 0:
                self._countdown = self._countdown - 1
                self._time = 0
            else:
                self._time = time - self._interval
                self._status = status

        return status, time
