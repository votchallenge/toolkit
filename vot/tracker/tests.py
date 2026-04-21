"""Unit tests for the tracker module."""

import unittest

import matplotlib.pylab as plt

from vot.utilities.draw import MatplotlibDrawHandle
from vot.dataset.dummy import generate_dummy
from vot.dataset import load_sequence, Frame
from vot.tracker import ObjectStatus, TrackerRuntime, \
    TrackerException, ObjectQuery, OnlineTrackerRuntime

from vot.experiment.helpers import MultiObjectHelper
from vot.dataset.proxy import ObjectsHideFilterSequence
from vot import get_logger
from vot.utilities import normalize_path

class TestStacks(unittest.TestCase):
    """Tests for the stacks module."""

    def test_trax_tracker_test(self):
        """Test tracker runtime with dummy sequence and dummy tracker."""
        from vot.tracker.dummy import DummyTraxTracker
        
        tracker = DummyTraxTracker
        sequence = generate_dummy(10)

        with tracker.runtime(log=False) as runtime:
            runtime.initialize(sequence.frame(0), [sequence.groundtruth(0)])
            for i in range(1, len(sequence)):
                runtime.update(sequence.frame(i))

    def test_folder_tracker_test(self):
        """Test folder tracker with dummy sequence and dummy tracker."""
        from vot.tracker.dummy import DummyFolderTracker
        
        tracker = DummyFolderTracker
        sequence = generate_dummy(10)

        with tracker.runtime(log=False) as runtime:
            queries = [ObjectQuery(sequence.groundtruth(0), {}, 0)]
            runtime.run(sequence, queries=queries)

def test_tracker_runtime(runtime: TrackerRuntime, visualize: bool = False, sequence: str = None, ignore: list = None):
    """Run a test for a tracker.

    :param config: Configuration
    :type config: argparse.Namespace
    """

    logger = get_logger()

    handle = None

    def visualize_state(axes, frame: Frame, reference, state):
        """Visualize the frame and the state of the tracker.

        :param axes: The axes to draw on.
        :type axes: matplotlib.axes.Axes
        :param frame: The frame to draw.
        :type frame: Frame
        :param reference: List of references.
        :type reference: list
        :param state: The state of the tracker.
        :type state: ObjectStatus
        """
        axes.clear()
        handle.image(frame.channel())
        if not isinstance(state, list):
            state = [state]
        for gt, st in zip(reference, state):
            handle.style(color="green").region(gt)
            handle.style(color="red").region(st.region)
        
    try:

        logger.info("Generating dummy sequence")

        if sequence is None:
            sequence = generate_dummy(50, objects=3 if runtime.multiobject else 1)
        else:
            sequence = load_sequence(normalize_path(sequence))

        if ignore:
            sequence = ObjectsHideFilterSequence(sequence, ignore)

        context = {"continue" : True}

        def on_press(event):
            """Callback for key press event.

            :param event: The event.
            :type event: matplotlib.backend_bases.Event
            """
            if event.key == 'q':
                context["continue"] = False

        if visualize:

            figure = plt.figure()
            if hasattr(figure.canvas, "set_window_title"):
                figure.canvas.set_window_title('VOT Test')
            axes = figure.add_subplot(1, 1, 1)
            axes.set_aspect("equal")
            handle = MatplotlibDrawHandle(axes, size=sequence.size)
            context["click"] = figure.canvas.mpl_connect('key_press_event', on_press)
            handle.style(fill=False)
            figure.show()

        helper = MultiObjectHelper(sequence)

        if isinstance(runtime, OnlineTrackerRuntime):

            logger.info("Initializing tracker")

            frame = sequence.frame(0)
            state, _ = runtime.initialize(frame, [ObjectStatus(frame.object(x), {}) for x in helper.new(0)])

            if visualize:
                visualize_state(axes, frame, [frame.object(x) for x in helper.objects(0)], state)
                figure.canvas.draw()

            for i in range(1, len(sequence)):
                
                logger.info(f"Processing frame {i}/{len(sequence)-1}")
                frame = sequence.frame(i)
                state, _ = runtime.update(frame, [ObjectStatus(frame.object(x), {}) for x in helper.new(i)])

                if visualize:
                    visualize_state(axes, frame, [frame.object(x) for x in helper.objects(i)], state)
                    figure.canvas.draw()
                    figure.canvas.flush_events()

                if not context["continue"]:
                    break

            logger.info("Stopping tracker")

            runtime.stop()

            logger.info("Test concluded successfuly")


        else:
            
            # Run tracker in batch mode
            
            queries = []
            for i in range(len(sequence)):
                frame = sequence.frame(i)
                queries.extend([ObjectQuery(frame.object(x), {}, i) for x in helper.new(i)])
            
            status = runtime.run(sequence, queries=queries)
            
            # Visualize results offline if requested
            if visualize:
                logger.info("Visualizing results")
                for i in range(len(sequence)):
                    frame = sequence.frame(i)
                    state = [obj[i] for obj in status.objects]
                    visualize_state(axes, frame, [frame.object(x) for x in helper.objects(i)], state)
                    figure.canvas.draw()
                    figure.canvas.flush_events()
                    if not context["continue"]:
                        break

    except TrackerException as te:
        logger.error(f"Error during tracker execution: {te}")
        if runtime:
            runtime.stop()
    except KeyboardInterrupt:
        if runtime:
            runtime.stop()