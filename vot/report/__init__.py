""" This module contains classes for generating reports and visualizations. """

import typing
import json
import inspect
import threading
import datetime
import collections
import collections.abc
import sys
from asyncio import wait, ensure_future
from asyncio.futures import wrap_future

import numpy as np
import yaml

from matplotlib.cm import get_cmap
from matplotlib.figure import Figure
from matplotlib.axes import Axes as PlotAxes
import matplotlib.colors as colors

from attributee import Attributee, Object, Nested, String, Callable, Integer, List

from vot import __version__ as version
from vot import get_logger
from vot.dataset import Sequence, FrameList
from vot.tracker import Tracker
from vot.analysis import Axes
#from vot.workspace import Storage, Workspace
from vot.utilities import class_fullname
from vot.utilities.data import Grid

Table = collections.namedtuple("Table", ["header", "data", "order"])

class Plot(object):
    """ Base class for all plots. """

    def __init__(self, identifier: str, xlabel: str, ylabel: str,
        xlimits: typing.Tuple[float, float], ylimits: typing.Tuple[float, float], trait = None):
        """ Initializes the plot.
        
        Args:
            identifier (str): The identifier of the plot.
            xlabel (str): The label of the x axis.
            ylabel (str): The label of the y axis.
            xlimits (tuple): The limits of the x axis.
            ylimits (tuple): The limits of the y axis.
            trait (str): The trait of the plot.    
        """

        self._identifier = identifier

        self._manager = StyleManager.default()

        self._figure, self._axes = self._manager.make_figure(trait)

        self._axes.xaxis.set_label_text(xlabel)
        self._axes.yaxis.set_label_text(ylabel)

        if not xlimits is None and not any([x is None for x in xlimits]):
            self._axes.set_xlim(xlimits)
            self._axes.autoscale(False, axis="x")
        if not ylimits is None and not any([y is None for y in ylimits]):
            self._axes.set_ylim(ylimits)
            self._axes.autoscale(False, axis="y")

    def __call__(self, key, data):
        """ Draws the data on the plot."""
        self.draw(key, data)

    def draw(self, key, data):
        """ Draws the data on the plot."""
        raise NotImplementedError
    
    @property
    def axes(self) -> Axes:
        """ Returns the axes of the plot."""
        return self._axes

    def save(self, output: str, fmt: str):
        """ Saves the plot to a file.
        
        Args:
            output (str): The output file.
            fmt (str): The format of the output file.
        """
        self._figure.savefig(output, format=fmt, bbox_inches='tight', transparent=True)

    @property
    def identifier(self):
        """ Returns the identifier of the plot."""
        return self._identifier

class Video(object):
    """ Base class for all videos. """

    def __init__(self, identifier: str, frames: FrameList, fps: int = 30, trait = None):
        """ Initializes the video object.
        
        Args:
            identifier (str): The identifier of the video.
            frames (FrameList): The frames of the video.
            fps (int): The frames per second of the video.
            trait (str): The trait of the video.    
        """

        self._identifier = identifier
        self._frames = frames
        self._fps = fps
        self._manager = StyleManager.default()

    def __call__(self, frame: int, key, data):
        """ Draws the data on the frame."""
        self.draw(frame, key, data)

    def draw(self, frame: int, key, data):
        """ Draws the data on the plot."""
        raise NotImplementedError

    def render(self, frame: int):
        """ Renders the frame and returns it as a NumPy array."""
        raise NotImplementedError

    def save(self, output: str, fmt: str):
        import tempfile
        import shutil
        import os
        from .video import VideoWriterScikitH264, VideoWriterOpenCV

        supported_mappings = {
            "mp4": VideoWriterScikitH264,
            "avi": VideoWriterOpenCV
        }

        if not fmt in supported_mappings:
            raise ValueError("Unsupported video format: {}".format(fmt))

        if not isinstance(output, str):
            fd, tempname = tempfile.mkstemp(prefix="video_", suffix=".{}".format(fmt))
            os.close(fd)
        else: 
            tempname = output

        writer = supported_mappings[fmt](tempname, self._fps)

        for i in range(0, len(self._frames)):
            writer(self.render(i))

        writer.close()

        if tempname == output:
            return
        
        shutil.copyfileobj(open(tempname, 'rb'), output)
        os.remove(tempname)

    def __len__(self):
        return len(self._frames)

    @property
    def identifier(self):
        """ Returns the identifier of the plot."""
        return self._identifier

class ScatterPlot(Plot):
    """ A scatter plot."""

    def draw(self, key, data):
        """ Draws the data on the plot. """
        if data is None or len(data) != 2:
            return

        style = self._manager.plot_style(key)
        self._axes.scatter(data[0], data[1], **style.point_style())

class LinePlot(Plot):
    """ A line plot."""

    def draw(self, key, data):
        """ Draws the data on the plot."""
        if data is None or len(data) < 1:
            return

        if isinstance(data[0], tuple):
            # Drawing curve
            if len(data[0]) != 2:
                return
            x, y = zip(*data)
        else:
            y = data
            x = range(len(data))

        style = self._manager.plot_style(key)

        self._axes.plot(x, y, **style.line_style())

class ObjectVideo(Video):

    def __init__(self, identifier: str, frames: FrameList, fps=10, trait=None):
        super().__init__(identifier, frames, fps=fps, trait=trait)
        self._regions = {}

    def draw(self, frame, key, data):
        """Draws the data on the frame."""
        from vot.region import Region
        assert isinstance(data, Region)

        if not key in self._regions:
            self._regions[key] = [None] * len(self)

        self._regions[key][frame] = data

    def render(self, frame: int):
        """Renders the frame and returns it as an array."""
        from vot.utilities.draw import ImageDrawHandle

        assert frame >= 0 and frame < len(self)

        handle = ImageDrawHandle(self._frames.frame(frame).image())

        for key, regions in self._regions.items():
            if regions[frame] is None:
                continue

            style = self._manager.plot_style(key)

            handle.style(**style.region_style())
            regions[frame].draw(handle)

        return handle.array


class ResultsJSONEncoder(json.JSONEncoder):
    """ JSON encoder for results. """

    def default(self, o):
        """ Default encoder. """
        if isinstance(o, Grid):
            return list(o)
        elif isinstance(o, datetime.date):
            return o.strftime('%Y/%m/%d')
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            return super().default(o)

class ResultsYAMLEncoder(yaml.Dumper):
    """ YAML encoder for results."""

    def represent_tuple(self, data):
        """ Represents a tuple. """
        return self.represent_list(list(data))


    def represent_object(self, o):
        """ Represents an object. """
        if isinstance(o, Grid):
            return self.represent_list(list(o))
        elif isinstance(o, datetime.date):
            return o.strftime('%Y/%m/%d')
        elif isinstance(o, np.ndarray):
            return self.represent_list(o.tolist())
        else:
            return super().represent_object(o)

ResultsYAMLEncoder.add_representer(collections.OrderedDict, ResultsYAMLEncoder.represent_dict)
ResultsYAMLEncoder.add_representer(tuple, ResultsYAMLEncoder.represent_tuple)
ResultsYAMLEncoder.add_representer(Grid, ResultsYAMLEncoder.represent_object)
ResultsYAMLEncoder.add_representer(np.ndarray, ResultsYAMLEncoder.represent_object)
ResultsYAMLEncoder.add_multi_representer(np.integer, ResultsYAMLEncoder.represent_int)
ResultsYAMLEncoder.add_multi_representer(np.inexact, ResultsYAMLEncoder.represent_float)

def generate_serialized(trackers: typing.List[Tracker], sequences: typing.List[Sequence], results, storage: "Storage", serializer: str, name: str):
    """ Generates a serialized report of the results.  """

    doc = dict()
    doc["toolkit"] = version
    doc["timestamp"] = datetime.datetime.now().isoformat()
    doc["trackers"] = {t.reference : t.describe() for t in trackers}
    doc["sequences"] = {s.name : s.describe() for s in sequences}

    doc["results"] = dict()

    for experiment, analyses in results.items():
        exp = dict(parameters=experiment.dump(), type=class_fullname(experiment))
        exp["results"] = []
        for _, data in analyses.items():
            exp["results"].append(data)
        doc["results"][experiment.identifier] = exp

    if serializer == "json":
        with storage.write(name + "." + serializer) as handle:
            json.dump(doc, handle, indent=2, cls=ResultsJSONEncoder)
    elif serializer == "yaml":
        with storage.write(name + "." + serializer) as handle:
            yaml.dump(doc, handle, Dumper=ResultsYAMLEncoder)
    else:
        raise RuntimeError("Unknown serializer")

def configure_axes(figure, rect=None, _=None):
    """ Configures the axes of the plot. """

    axes = PlotAxes(figure, rect or [0, 0, 1, 1])

    figure.add_axes(axes)

    return axes

def configure_figure(traits=None):
    """ Configures the figure of the plot. """

    args = {}
    if traits == "ar":
        args["figsize"] = (5, 5)
    elif traits == "eao":
        args["figsize"] = (7, 5)
    elif traits == "attributes":
        args["figsize"] = (10, 5)

    return Figure(**args)

class PlotStyle(object):
    """ A style for a plot."""

    def line_style(self, opacity=1):
        """ Returns the style for a line."""
        raise NotImplementedError

    def point_style(self):
        """ Returns the style for a point."""
        raise NotImplementedError

    def region_style(self):
        """ Returns the style for a region, used with DrawHandle."""
        raise NotImplementedError

class DefaultStyle(PlotStyle):
    """ The default style for a plot."""

    colormap = get_cmap("tab20b")
    colorcount = 20
    markers = ["o", "v", "<", ">", "^", "8", "*"]

    def __init__(self, number):
        """ Initializes the style. 
        
        Args:
            number (int): The number of the style.
        """
        super().__init__()
        self._number = number

    def line_style(self, opacity=1):
        """ Returns the style for a line.
        
        Args:
            opacity (float): The opacity of the line.
        """
        color = DefaultStyle.colormap((self._number % DefaultStyle.colorcount + 1) / DefaultStyle.colorcount)
        if opacity < 1:
            color = colors.to_rgba(color, opacity)
        return dict(linewidth=1, c=color)

    def point_style(self):
        """ Returns the style for a point.
        
        Args:
            color (str): The color of the point.
            opacity (float): The opacity of the line.
        """
        color = DefaultStyle.colormap((self._number % DefaultStyle.colorcount + 1) / DefaultStyle.colorcount)
        marker = DefaultStyle.markers[self._number % len(DefaultStyle.markers)]
        return dict(marker=marker, c=[color])

    def region_style(self):
        """ Returns the style for a region, used with DrawHandle."""
        color = DefaultStyle.colormap((self._number % DefaultStyle.colorcount + 1) / DefaultStyle.colorcount)
        return dict(color=color, fill=True)

class Legend(object):
    """ A legend for a plot."""

    def __init__(self, style_factory=DefaultStyle):
        """ Initializes the legend.
        
        Args:
            style_factory (PlotStyleFactory): The style factory.
        """
        self._mapping = collections.OrderedDict()
        self._counter = 0
        self._style_factory = style_factory

    def _number(self, key):
        """ Returns the number for a key."""
        if not key in self._mapping:
            self._mapping[key] = self._counter
            self._counter += 1
        return self._mapping[key]

    def __getitem__(self, key) -> PlotStyle:
        """ Returns the style for a key."""
        number = self._number(key)
        return self._style_factory(number)

    def _style(self, number):
        """ Returns the style for a number."""
        raise NotImplementedError

    def keys(self):
        """ Returns the keys of the legend."""
        return self._mapping.keys()

    def figure(self, key):
        """ Returns a figure for a key."""
        style = self[key]
        figure = Figure(figsize=(0.1, 0.1))  # TODO: hardcoded
        axes = PlotAxes(figure, [0, 0, 1, 1], yticks=[], xticks=[], frame_on=False)
        figure.add_axes(axes)
        axes.patch.set_visible(False)
        marker_style = style.point_style()
        marker_style["s"] = 40 # Reset size
        axes.scatter(0, 0, **marker_style)
        return figure

class StyleManager(Attributee):
    """ A manager for styles. """

    plots = Callable(default=DefaultStyle)
    axes = Callable(default=configure_axes)
    figure = Callable(default=configure_figure)

    _context = threading.local()

    def __init__(self, **kwargs):
        """ Initializes a new instance of the StyleManager class."""
        super().__init__(**kwargs)
        self._legends = dict()

    def __getitem__(self, key) -> PlotStyle:
        """ Gets the style for the given key."""
        return self.plot_style(key)

    def legend(self, key) -> Legend:
        """ Gets the legend for a given key."""
        if inspect.isclass(key):
            klass = key
        else:
            klass = type(key)

        if not klass in self._legends:
            self._legends[klass] = Legend(self.plots)

        return self._legends[klass]

    def plot_style(self, key) -> PlotStyle:
        """ Gets the plot style for a given key."""
        return self.legend(key)[key]

    def make_axes(self, figure, rect=None, trait=None) -> Axes:
        """ Makes the axes for a given figure."""
        return self.axes(figure, rect, trait)

    def make_figure(self, trait=None) -> typing.Tuple[Figure, Axes]:
        """ Makes the figure for a given trait.
        
        Args:
            trait (str): The trait for which to make the figure.

        Returns:
            A tuple containing the figure and the axes.
        """
        figure = self.figure(trait)
        axes = self.make_axes(figure, trait=trait)

        return figure, axes

    def __enter__(self):
        """Enters the context of the style manager."""

        manager = getattr(StyleManager._context, 'style_manager', None)

        if manager == self:
            return self

        StyleManager._context.style_manager = self

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exits the context of the style manager."""
        manager = getattr(StyleManager._context, 'style_manager', None)

        if manager == self:
            StyleManager._context.style_manager = None

    @staticmethod
    def default() -> "StyleManager":
        """ Gets the default style manager."""

        manager = getattr(StyleManager._context, 'style_manager', None)
        if manager is None:
            get_logger().info("Creating new style manager", stack_info=True)
            manager = StyleManager()
            StyleManager._context.style_manager = manager

        return manager

class TrackerSorter(Attributee):
    """ A sorter for trackers. """

    experiment = String(default=None)
    analysis = String(default=None)
    result = Integer(val_min=0, default=0)

    def __call__(self, experiments: typing.List["Experiment"], trackers: typing.List["Tracker"], sequences: typing.List["Sequence"]):
        """ Sorts the trackers. 
        
        Arguments:
            experiments (typing.List[Experiment]): The experiments.
            trackers (typing.List[Tracker]): The trackers.
            sequences (typing.List[Sequence]): The sequences.
            
        Returns:
            A list of indices of the trackers in the sorted order.
        """
        from vot.analysis import AnalysisError

        if self.experiment is None or self.analysis is None:
            return range(len(trackers))

        experiment = next(filter(lambda x: x.identifier == self.experiment, experiments), None)

        if experiment is None:
            raise RuntimeError("Experiment not found")

        analysis = next(filter(lambda x: x.name == self.analysis, experiment.analyses), None)

        if analysis is None:
            raise RuntimeError("Analysis not found")

        try:

            future = analysis.commit(experiment, trackers, sequences)
            result = future.result()
        except AnalysisError as e:
            raise RuntimeError("Unable to sort trackers", e)

        scores = [x[self.result] for x in result]
        indices = [i[0] for i in sorted(enumerate(scores), reverse=True, key=lambda x: x[1])]

        return indices

class Report(Attributee):
    """ A report generator for various reports. Base class for all report generators. """

    async def generate(self, experiments, trackers, sequences):
        raise NotImplementedError()

    async def process(self, analyses, experiment, trackers, sequences):

        sequences = experiment.transform(sequences)

        if sys.version_info >= (3, 3):
            _Iterable = collections.abc.Iterable
        else:
            _Iterable = collections.Iterable
        if not isinstance(analyses, _Iterable):
            analyses = [analyses]

        futures = []

        for analysis in analyses:
            futures.append(wrap_future(analysis.commit(experiment, trackers, sequences)))

        await wait(futures)

        return (future.result() for future in futures)

class SeparableReport(Report):
    """ A report generator that is separable across experiments. Base class for all separable report generators. """

    async def perexperiment(self, experiment, trackers, sequences):
        raise NotImplementedError()

    def compatible(self, experiment):
        raise NotImplementedError()

    async def generate(self, experiments, trackers, sequences):

        futures = []
        texperiments = []

        for experiment in experiments:

            tsequences = experiment.transform(sequences)

            if self.compatible(experiment):
                futures.append(ensure_future(self.perexperiment(experiment, trackers, tsequences)))
                texperiments.append(experiment)
            else:
                continue

        await wait(futures)

        items = dict()

        for experiment, future in zip(texperiments, futures):
            items[experiment.identifier] = future.result()

        return items



class ReportConfiguration(Attributee):
    """ A configuration for reports."""

    style = Nested(StyleManager)
    sort = Nested(TrackerSorter)
    index = List(Object(subclass=Report), default=[], description="The reports to include.")

def generate_document(workspace: "Workspace", trackers: typing.List[Tracker], format: str, name: str, select_sequences: typing.Optional[str] = None, select_experiments: typing.Optional[str] = None):
    """Generate a report for a one or multiple trackers on an experiment stack and a set of sequences.

    Args:
        workspace (Workspace): The workspace to use for the report.
        trackers: The trackers to include in the report.
        format: The format of the report.
        name: The name of the report.
    """
    from asyncio import ensure_future, get_event_loop, wait

    from vot.analysis import AnalysisProcessor
    from vot.utilities import Progress
    from vot.workspace.storage import Cache
    from vot import config
    from vot.report.common import StackAnalysesTable, StackAnalysesPlots

    def merge_tree(src, dest):

        for key, value in src.items():
            if not key in dest:        
                dest[key] = list()
            dest[key] += value

    logger = get_logger()

    if config.worker_pool_size == 1:

        if config.debug_mode:
            import logging
            from vot.analysis.processor import DebugExecutor
            logging.getLogger("concurrent.futures").setLevel(logging.DEBUG)
            executor = DebugExecutor()
        else:
            from vot.utilities import ThreadPoolExecutor
            executor = ThreadPoolExecutor(1)

    else:
        from concurrent.futures import ProcessPoolExecutor
        executor = ProcessPoolExecutor(config.workers)

    if not config.persistent_cache:
        from cachetools import LRUCache
        cache = LRUCache(1000)
    else:
        cache = Cache(workspace.storage.substorage("cache").substorage("analysis"))

    index = workspace.report.index
    if len(index) == 0:
        # Default report content
        index = [StackAnalysesTable(), StackAnalysesPlots()]
        
    with workspace.report.style:

        experiments = workspace.stack
        sequences = workspace.dataset
        
        if not select_experiments is None:
            experiments = [experiment for name, experiment in workspace.stack.experiments.items() if name in select_experiments.split(",")]
        if not select_sequences is None:
            sequences = [sequence for sequence in sequences if sequence.name in select_sequences.split(",")]


        if len(experiments) == 0:
            logger.warning("No experiments selected")

        if len(sequences) == 0:
            logger.warning("No sequences selected")

        try:

            with AnalysisProcessor(executor, cache) as processor:
                
                order = workspace.report.sort(experiments, trackers, sequences)

                trackers = [trackers[i] for i in order]

                futures = []

                for report in index:
                    futures.append(ensure_future(report.generate(experiments, trackers, sequences)))

                loop = get_event_loop()

                progress = Progress("Processing", processor.total)

                def update():
                    progress.total(processor.total)
                    progress.absolute(processor.total - processor.pending)
                    loop.call_later(1, update)

                update()

                if len(futures) > 0:
                    loop.run_until_complete(wait(futures))

                progress.close()

                reports = dict()

                for future in futures:
                    merge_tree(future.result(), reports)

        finally:

            executor.shutdown(wait=True)

        report_storage = workspace.storage.substorage("reports").substorage(name)

        def only_plots(reports, storage: "Storage"):
            """Filter out all non-plot items from the report and save them to storage.
            
            Args:
                reports: The reports to filter.
            """
            for key, section in reports.items():
                for item in section:
                    if isinstance(item, Plot):
                        logger.debug("Saving plot %s", item.identifier)
                        with storage.write(key + "_" + item.identifier + '.pdf', binary=True) as out:
                            item.save(out, "PDF")
                        with storage.write(key + "_" + item.identifier + '.png', binary=True) as out:
                            item.save(out, "PNG")
                    if isinstance(item, Video):
                        logger.debug("Saving video %s", item.identifier)
                        with storage.write(key + "_" + item.identifier + '.avi', binary=True) as out:
                            item.save(out, "avi")

        metadata = {"Stack": workspace.stack.title}

        # Prune empty sections
        reports = {key: section for key, section in reports.items() if len(section) > 0}

        if format == "html":
            from .html import generate_html_document
            generate_html_document(trackers, workspace.dataset, reports, report_storage, metadata=metadata)
        elif format == "latex":
            from .latex import generate_latex_document
            generate_latex_document(trackers, workspace.dataset, reports, report_storage, metadata=metadata)
        elif format == "plots":
            only_plots(reports, report_storage)
        else:
            raise ValueError("Unknown report format %s" % format)